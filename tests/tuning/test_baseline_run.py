"""Baseline run orchestrator (Stage 7 step 2A): synthetic WAV files, caches and manifest, seed 2026.

Convention: label 1 = spoof (FAKE), 0 = bonafide (REAL).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import src.tuning.baseline_fit as fit_module
import src.tuning.baseline_run as run_module
from src.evaluation.metrics import eer
from src.features.mfcc_cache import build_mfcc_cache
from src.models.baseline import artifact_sha256, load
from src.preprocessing import PreprocessingError, load_profile

REPO = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO / "configs" / "preprocessing.yaml"
CONFIG_PATH = REPO / "configs" / "baseline.yaml"
COMMIT = "c" * 40
# Rows per subgroup (REAL, FAKE_mp3, FAKE_other); different split sizes let a spy tell them apart.
SIZES = {"train": (6, 3, 3), "val-A": (4, 2, 3), "val-B": (3, 2, 2)}
OUTPUTS = sorted(["artifact", "run_report.json", "selection_report.json",
                  "subgroup_report.json", "threshold_report.json"])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def profile():
    return load_profile(PROFILE_PATH)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory, profile):
    """Synthetic FoR-like corpus, its manifest CSV and the three caches built from it."""
    base = tmp_path_factory.mktemp("baseline_run")
    rng = np.random.default_rng(2026)
    root = base / "for"
    rows = []
    specs = [(split, sub) for split, counts in SIZES.items()
             for sub, n in zip(("REAL", "FAKE_mp3", "FAKE_other"), counts) for _ in range(n)]
    specs.append(("test", "REAL"))  # in the manifest only; the run never reads it
    for i, (split, subgroup) in enumerate(specs):
        rate = 16000
        t = np.arange(int(rate * rng.uniform(1.4, 2.4))) / rate
        tone = np.sin(2 * np.pi * rng.uniform(150, 900) * t)
        noise = rng.standard_normal(len(t))
        mix = {"REAL": 0.1, "FAKE_mp3": 0.5, "FAKE_other": 0.8}[subgroup] + rng.uniform(-0.1, 0.1)
        data = 0.3 * ((1 - mix) * tone + mix * noise)
        rel = f"{split}/{subgroup}/{i:02d}.wav"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, data.astype(np.float32), rate, subtype="PCM_16")
        rows.append({"path": rel, "label": "REAL" if subgroup == "REAL" else "FAKE",
                     "subgroup": subgroup, "group": f"g{i}", "split": split, "hash": _sha(path)})
    manifest = pd.DataFrame(rows, dtype=str)
    manifest_path = base / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    cache_dir = base / "cache"
    for split in SIZES:
        build_mfcc_cache(root, manifest, profile, split, cache_dir / split,
                         f2_manifest_sha256=_sha(manifest_path),
                         profile_config_sha256=_sha(PROFILE_PATH), code_commit=COMMIT)
    return {"cache_dir": cache_dir, "manifest_path": manifest_path, "manifest": manifest}


@pytest.fixture()
def patched(inputs, monkeypatch):
    """Clean worktree, and the synthetic manifest standing in for the approved F2 file."""
    monkeypatch.setattr(run_module, "code_version", lambda: {"git_commit": COMMIT, "git_dirty": False})
    monkeypatch.setattr(run_module, "load_f2_manifest",
                        lambda p: pd.read_csv(p, dtype=str, keep_default_na=False))
    monkeypatch.setattr(run_module, "F2_MANIFEST_SHA256", _sha(inputs["manifest_path"]))
    return inputs


def _run(inputs, out, cache_dir=None):
    return run_module.run(cache_dir or inputs["cache_dir"], PROFILE_PATH, CONFIG_PATH,
                          inputs["manifest_path"], out, "baseline-test")


def test_end_to_end_outputs_keys_and_hashes(patched, tmp_path, profile):
    out = tmp_path / "run"
    report = _run(patched, out)
    assert sorted(p.name for p in out.iterdir()) == OUTPUTS
    data = json.loads((out / "run_report.json").read_text(encoding="utf-8"))
    assert data == report and list(data) == sorted(data)
    assert set(data) == {
        "schema_version", "version", "code_commit", "profile_fingerprint", "profile_config_sha256",
        "f2_manifest_sha256", "baseline_config_sha256", "caches", "artifact_sha256", "selected_C",
        "selected_gamma", "val_a_eer", "val_b_threshold", "val_b_eer",
    }
    assert (data["schema_version"], data["version"], data["code_commit"]) == (1, "baseline-test", COMMIT)
    assert data["artifact_sha256"] == artifact_sha256(out / "artifact")
    assert data["profile_fingerprint"] == profile.fingerprint()
    assert data["profile_config_sha256"] == _sha(PROFILE_PATH)
    assert data["baseline_config_sha256"] == _sha(CONFIG_PATH)
    assert data["f2_manifest_sha256"] == _sha(patched["manifest_path"])
    assert set(data["caches"]) == set(SIZES)
    for split, entry in data["caches"].items():
        meta = json.loads((patched["cache_dir"] / split / f"{split}.json").read_text(encoding="utf-8"))
        assert entry == {"npz_sha256": meta["npz_sha256"], "n_rows": meta["n_rows"]}
        assert entry["npz_sha256"] == _sha(patched["cache_dir"] / split / f"{split}.npz")

    selection = json.loads((out / "selection_report.json").read_text(encoding="utf-8"))
    threshold = json.loads((out / "threshold_report.json").read_text(encoding="utf-8"))
    assert set(selection) == {"rows", "selected_C", "selected_gamma", "n_train", "n_val_a",
                              "sklearn_version", "svc_settings"}
    assert set(threshold) == {"threshold", "val_b_eer", "n_val_b", "n_spoof_val_b",
                              "n_bonafide_val_b"}
    assert len(selection["rows"]) == 16
    assert (data["selected_C"], data["selected_gamma"]) == (selection["selected_C"],
                                                            selection["selected_gamma"])
    assert data["val_a_eer"] == min(r["val_a_eer"] for r in selection["rows"])
    assert (data["val_b_threshold"], data["val_b_eer"]) == (threshold["threshold"],
                                                            threshold["val_b_eer"])
    model = load(out / "artifact")
    assert (model.version, model.threshold) == ("baseline-test", data["val_b_threshold"])
    subgroup = json.loads((out / "subgroup_report.json").read_text(encoding="utf-8"))
    assert set(subgroup) == {"purpose", "splits"}


def test_two_runs_are_byte_identical(patched, tmp_path):
    first = _run(patched, tmp_path / "a")
    second = _run(patched, tmp_path / "b")
    assert first["artifact_sha256"] == second["artifact_sha256"]
    files = sorted(p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*") if p.is_file())
    assert len(files) == 6  # four reports + baseline.npz + baseline.json
    for rel in files:
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes()


def _assert_nothing_written(out: Path) -> None:
    assert not out.exists()


@pytest.mark.parametrize("split", sorted(SIZES))
def test_missing_cache_is_refused(patched, tmp_path, split):
    cache_dir = tmp_path / "cache"
    shutil.copytree(patched["cache_dir"], cache_dir)
    shutil.rmtree(cache_dir / split)
    with pytest.raises(ValueError, match="missing cache file"):
        _run(patched, tmp_path / "out", cache_dir=cache_dir)
    _assert_nothing_written(tmp_path / "out")


def test_existing_out_dir_is_refused(patched, tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    calls = []
    monkeypatch.setattr(run_module, "load_mfcc_cache", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="already exists"):
        _run(patched, out)
    assert calls == [] and list(out.iterdir()) == []


def test_dirty_worktree_is_refused_before_reading(inputs, tmp_path, monkeypatch):
    def dirty():
        raise PreprocessingError("requires a clean Git worktree")
    calls = []
    monkeypatch.setattr(run_module, "code_version", dirty)
    for name in ("load_profile", "load_baseline_config", "load_f2_manifest", "load_mfcc_cache"):
        monkeypatch.setattr(run_module, name, lambda *a, _n=name, **k: calls.append(_n))
    with pytest.raises(PreprocessingError, match="clean Git worktree"):
        _run(inputs, tmp_path / "out")
    assert calls == []
    _assert_nothing_written(tmp_path / "out")


def test_cache_built_with_another_profile_is_refused(patched, tmp_path, monkeypatch, profile):
    other = dataclasses.replace(profile, features={**profile.features, "fmax_hz": 8000.0})
    assert other.fingerprint() != profile.fingerprint()
    monkeypatch.setattr(run_module, "load_profile", lambda p: other)
    with pytest.raises(ValueError, match="profile fingerprint check failed"):
        _run(patched, tmp_path / "out")
    _assert_nothing_written(tmp_path / "out")


def test_failure_after_fit_removes_out_dir(patched, tmp_path, monkeypatch):
    def broken(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(run_module, "artifact_sha256", broken)
    with pytest.raises(OSError, match="disk full"):
        _run(patched, tmp_path / "out")
    _assert_nothing_written(tmp_path / "out")


def test_cache_path_missing_from_manifest_raises(patched, tmp_path, monkeypatch):
    manifest = patched["manifest"]
    dropped = manifest[manifest["path"] != manifest.loc[manifest.split == "val-B", "path"].iloc[0]]
    monkeypatch.setattr(run_module, "load_f2_manifest", lambda p: dropped.reset_index(drop=True))
    with pytest.raises(ValueError, match="not in the F2 manifest"):
        _run(patched, tmp_path / "out")
    _assert_nothing_written(tmp_path / "out")


def test_selection_uses_val_a_only_and_threshold_val_b_only(patched, tmp_path, monkeypatch):
    sizes = {split: sum(counts) for split, counts in SIZES.items()}
    calls, loaded = [], []

    def spy(labels, scores):
        calls.append(len(labels))
        return eer(labels, scores)

    original_load = run_module.load_mfcc_cache

    def load_spy(path, split, *a, **k):
        loaded.append(split)
        return original_load(path, split, *a, **k)

    monkeypatch.setattr(fit_module, "eer", spy)
    monkeypatch.setattr(run_module, "load_mfcc_cache", load_spy)
    _run(patched, tmp_path / "out")
    # 16 selection calls see only val-A labels, then the threshold call sees only val-B labels.
    assert calls == [sizes["val-A"]] * 16 + [sizes["val-B"]]
    assert loaded == ["train", "val-A", "val-B"]


def test_subgroup_report_is_descriptive_and_counts_add_up(patched, tmp_path):
    out = tmp_path / "out"
    report = _run(patched, out)
    subgroup = json.loads((out / "subgroup_report.json").read_text(encoding="utf-8"))
    assert subgroup["purpose"] == run_module.SUBGROUP_PURPOSE
    assert subgroup["purpose"].startswith("descriptive only; not used for selection")
    assert set(subgroup["splits"]) == {"val-A", "val-B"}
    for split, entry in subgroup["splits"].items():
        n_real, n_mp3, n_other = SIZES[split]
        assert entry["counts"] == {"REAL": n_real, "FAKE_mp3": n_mp3, "FAKE_other": n_other}
        assert sum(entry["counts"].values()) == entry["n_rows"] == report["caches"][split]["n_rows"]
        assert entry["REAL_vs_FAKE_mp3"]["n_bonafide"] == entry["REAL_vs_FAKE_other"]["n_bonafide"]
        assert entry["REAL_vs_FAKE_mp3"]["n_spoof"] + entry["REAL_vs_FAKE_other"]["n_spoof"] \
            == entry["n_rows"] - entry["counts"]["REAL"]
        for fake in ("FAKE_mp3", "FAKE_other"):
            assert 0.0 <= entry[f"REAL_vs_{fake}"]["eer"] <= 1.0


def test_cli_has_no_split_or_test_option(patched, tmp_path):
    with pytest.raises(SystemExit):
        run_module.main(["--cache-dir", "c", "--profile", "p", "--config", "c", "--manifest", "m",
                         "--out", "o", "--version", "v", "--split", "test"])
    run_module.main(["--cache-dir", str(patched["cache_dir"]), "--profile", str(PROFILE_PATH),
                     "--config", str(CONFIG_PATH), "--manifest", str(patched["manifest_path"]),
                     "--out", str(tmp_path / "out"), "--version", "baseline-cli"])
    assert json.loads((tmp_path / "out" / "run_report.json").read_text())["version"] == "baseline-cli"
