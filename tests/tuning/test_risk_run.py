"""Stage 10 risk-policy run: checkpoint binding, val-A reproduction, val-B fit, outputs, diagnostics.

Synthetic WAV files, manifest and log-Mel caches, and a tiny CNN trained on the synthetic train
cache with train_one (seed 2026 data, seed 0 training); CPU; no W&B.
Convention: label 1 = spoof (FAKE), 0 = bonafide (REAL).
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from tests._torch import require_torch

torch = require_torch()

import src.tuning.risk_run as run_module  # noqa: E402
from src.features.logmel_cache import build_logmel_cache, load_logmel_cache  # noqa: E402
from src.models.cnn import build_cnn  # noqa: E402
from src.models.cnn_checkpoint import checkpoint_sha256, save_checkpoint  # noqa: E402
from src.preprocessing import load_profile  # noqa: E402
from src.risk.calibration import PlattCalibration  # noqa: E402
from src.risk.policy import RiskPolicy  # noqa: E402
from src.training.train_cnn import train_one  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO / "configs" / "preprocessing.yaml"
CONFIG_PATH = REPO / "configs" / "cnn.yaml"
RUN_FILE = REPO / "src" / "tuning" / "risk_run.py"
COMMIT = "c" * 40
# Rows per subgroup (REAL, FAKE_mp3, FAKE_other). train is cached but must never be read here.
SIZES = {"train": (8, 4, 4), "val-A": (4, 2, 2), "val-B": (4, 3, 3)}
OUTPUTS = sorted(["diagnostics_report.json", "risk_policy.json", "run_report.json",
                  "val_b_scores.npz"])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _NullLogger:
    def log_epoch(self, *args):
        pass

    def log_summary(self, *args):
        pass


@pytest.fixture(scope="module")
def profile():
    return load_profile(PROFILE_PATH)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory, profile):
    """Synthetic corpus, manifest, train/val-A/val-B caches and a tiny trained checkpoint (seed 0)."""
    base = tmp_path_factory.mktemp("risk_run")
    rng = np.random.default_rng(2026)
    root = base / "for"
    rows = []
    specs = [(split, sub) for split, counts in SIZES.items()
             for sub, n in zip(("REAL", "FAKE_mp3", "FAKE_other"), counts) for _ in range(n)]
    specs.append(("test", "REAL"))  # in the manifest only; the run never reads it
    for i, (split, subgroup) in enumerate(specs):
        rate = 16000
        t = np.arange(int(rate * rng.uniform(1.6, 2.2))) / rate
        tone = np.sin(2 * np.pi * rng.uniform(150, 900) * t)
        mix = {"REAL": 0.1, "FAKE_mp3": 0.6, "FAKE_other": 0.8}[subgroup] + rng.uniform(-0.05, 0.05)
        data = 0.3 * ((1 - mix) * tone + mix * rng.standard_normal(len(t)))
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
        build_logmel_cache(root, manifest, profile, split, cache_dir / split,
                           f2_manifest_sha256=_sha(manifest_path),
                           profile_config_sha256=_sha(PROFILE_PATH), code_commit=COMMIT)

    # A real (tiny) training run, so the checkpoint's val_a_eer is the value train_one recorded,
    # and the runner must reproduce it through its own scoring path.
    train_x, train_y, _, _ = load_logmel_cache(cache_dir / "train", "train", profile,
                                               expected_f2_manifest_sha256=_sha(manifest_path))
    val_a_x, val_a_y, _, _ = load_logmel_cache(cache_dir / "val-A", "val-A", profile,
                                               expected_f2_manifest_sha256=_sha(manifest_path))
    result = train_one(train_x, train_y, val_a_x, val_a_y, n_blocks=3, learning_rate=0.001, seed=0,
                       max_epochs=4, patience=4, batch_size=8, logger=_NullLogger(), device="cpu")
    model = build_cnn(3)
    model.load_state_dict(result.state_dict, strict=True)

    def checkpoint(name, **changes):
        meta = {
            "schema_version": 1, "version": "cnn-test", "n_blocks": 3, "channels": [16, 32, 64],
            "dropout": 0.3, "input_shape": [1, 64, 197], "learning_rate": 0.001, "seed": 0,
            "best_epoch": result.best_epoch, "val_a_eer": result.best_val_a_eer,
            "code_commit": COMMIT, "environment": {"python": "3.13.15"},
            "profile_fingerprint": profile.fingerprint(), "f2_manifest_sha256": _sha(manifest_path),
            "train_cache_npz_sha256": "d" * 64, "val_a_cache_npz_sha256": "e" * 64,
            "determinism": dict(result.determinism),
        }
        meta.update(changes)
        if "seed" in changes:
            meta["determinism"]["seed"] = changes["seed"]
        folder = save_checkpoint(model, result.mean, result.std, meta, base / "ckpt" / name)
        return folder, checkpoint_sha256(folder)

    good, good_sha = checkpoint("good")
    return {"base": base, "cache_dir": cache_dir, "manifest_path": manifest_path,
            "manifest": manifest, "checkpoint": checkpoint, "good": good, "good_sha": good_sha,
            "evidence": _evidence(base / "evidence_good", 0, {"0": good_sha, "1": "1" * 64})}


def _evidence(folder: Path, bundle_seed: int, hashes: dict, selection_seed=None) -> Path:
    """A minimal Stage 8 evidence folder: run_report (bundle seed, hashes) and selection_report."""
    folder.mkdir(parents=True)
    (folder / "run_report.json").write_text(
        json.dumps({"bundle_seed": bundle_seed, "checkpoint_sha256": hashes}), encoding="utf-8")
    seed = bundle_seed if selection_seed is None else selection_seed
    (folder / "selection_report.json").write_text(json.dumps({"bundle_seed": seed}), encoding="utf-8")
    return folder


@pytest.fixture()
def patched(inputs, monkeypatch):
    """Clean worktree, the synthetic manifest as the approved F2 file, and a cache-load spy."""
    monkeypatch.setattr(run_module, "code_version", lambda: {"git_commit": COMMIT, "git_dirty": False})
    monkeypatch.setattr(run_module, "load_f2_manifest",
                        lambda p: pd.read_csv(p, dtype=str, keep_default_na=False))
    monkeypatch.setattr(run_module, "F2_MANIFEST_SHA256", _sha(inputs["manifest_path"]))
    loaded = []
    real_load = run_module.load_logmel_cache

    def spy(out_dir, split, *args, **kw):
        loaded.append(split)
        return real_load(out_dir, split, *args, **kw)

    monkeypatch.setattr(run_module, "load_logmel_cache", spy)
    return {**inputs, "loaded": loaded}


def _run(inputs, out, checkpoint=None, evidence=None):
    return run_module.run(inputs["cache_dir"], checkpoint or inputs["good"], PROFILE_PATH, CONFIG_PATH,
                          inputs["manifest_path"], evidence or inputs["evidence"], out,
                          version="risk-test")


def test_end_to_end_outputs_round_trip_and_determinism(patched, tmp_path, profile):
    # One run writes exactly the four files; a second run into a new folder gives the same bytes,
    # because scoring is on CPU, the fit is deterministic and every writer is byte-stable.
    out = tmp_path / "run"
    report = _run(patched, out)
    assert sorted(p.name for p in out.iterdir()) == OUTPUTS

    data = json.loads((out / "run_report.json").read_text(encoding="utf-8"))
    assert data == report and list(data) == sorted(data)
    assert set(data) == {
        "schema_version", "version", "code_commit", "environment", "profile_fingerprint",
        "profile_config_sha256", "f2_manifest_sha256", "caches", "bundle_seed", "checkpoint_sha256",
        "n_blocks", "scoring", "val_a_eer_recomputed", "risk_policy_sha256", "val_b_scores_sha256",
        "headline",
    }
    assert (data["schema_version"], data["version"], data["code_commit"]) == (1, "risk-test", COMMIT)
    assert set(data["environment"]) == {"python", "numpy", "torch"}
    assert (data["bundle_seed"], data["checkpoint_sha256"], data["n_blocks"]) == (0, patched["good_sha"], 3)
    assert data["scoring"] == {"device": "cpu", "batch_size": 256}  # from configs/cnn.yaml
    assert data["profile_fingerprint"] == profile.fingerprint()
    assert data["profile_config_sha256"] == _sha(PROFILE_PATH)
    assert data["f2_manifest_sha256"] == _sha(patched["manifest_path"])
    assert set(data["caches"]) == {"val-A", "val-B"}
    for split, entry in data["caches"].items():
        npz = patched["cache_dir"] / split / f"{split}.npz"
        assert entry == {"npz_sha256": _sha(npz), "n_rows": sum(SIZES[split])}
    assert data["risk_policy_sha256"] == _sha(out / "risk_policy.json")
    assert data["val_b_scores_sha256"] == _sha(out / "val_b_scores.npz")

    policy_text = (out / "risk_policy.json").read_text(encoding="utf-8")
    policy = RiskPolicy.from_json(policy_text)
    assert policy.to_json() == policy_text
    headline = data["headline"]
    assert headline == {"a": policy.calibration.a, "b": policy.calibration.b,
                        "threshold": policy.threshold, "half_width": policy.half_width,
                        "deferral_on_fit": policy.deferral_on_fit, "n_bonafide": 4, "n_spoof": 6}

    with np.load(io.BytesIO((out / "val_b_scores.npz").read_bytes()), allow_pickle=False) as npz:
        assert sorted(npz.files) == ["labels", "p", "paths", "score"]
        paths, labels, score, p = npz["paths"], npz["labels"], npz["score"], npz["p"]
    manifest = patched["manifest"]
    assert paths.tolist() == sorted(manifest.loc[manifest.split == "val-B", "path"])
    assert labels.dtype == np.int8 and labels.tolist() == [1] * 6 + [0] * 4  # "val-B/FAKE_*" sorts before "val-B/REAL"
    assert score.dtype == np.float64 and p.dtype == np.float64
    np.testing.assert_array_equal(p, policy.calibration.transform(score))

    diag = json.loads((out / "diagnostics_report.json").read_text(encoding="utf-8"))
    assert diag == json.loads(json.dumps(run_module.diagnostics(labels, score, policy,
                                                                np.array([s.split("/")[1] for s in paths]))))

    again = tmp_path / "again"
    _run(patched, again)
    for name in OUTPUTS:
        assert (again / name).read_bytes() == (out / name).read_bytes(), name


def test_caches_loaded_are_val_a_then_val_b_only(patched, tmp_path):
    # train/ exists in cache_dir and the manifest has a test row, but neither may be read (D9, D31).
    _run(patched, tmp_path / "run")
    assert patched["loaded"] == ["val-A", "val-B"]


def test_checkpoint_sha_mismatch_is_refused_before_any_cache_load(patched, tmp_path):
    # The evidence records another hash for seed 0, so load_checkpoint raises before any cache read.
    evidence = _evidence(tmp_path / "ev", 0, {"0": "0" * 64})
    with pytest.raises(ValueError, match="checkpoint_sha256 mismatch"):
        _run(patched, tmp_path / "out", evidence=evidence)
    assert patched["loaded"] == []
    assert not (tmp_path / "out").exists()


def test_seed_mismatch_is_refused_before_any_cache_load(patched, tmp_path):
    # A seed-1 checkpoint whose hash is listed under the bundle seed 0 passes the hash check but is
    # not the bundle seed (D45), so it is refused before any cache read.
    folder, sha = patched["checkpoint"]("seed1", seed=1)
    evidence = _evidence(tmp_path / "ev", 0, {"0": sha})
    with pytest.raises(ValueError, match="not the Stage 8 bundle seed 0"):
        _run(patched, tmp_path / "out", checkpoint=folder, evidence=evidence)
    assert patched["loaded"] == []


def test_inconsistent_stage8_evidence_is_refused(patched, tmp_path):
    # run_report says bundle seed 0, selection_report says 1: which checkpoint to use is ambiguous.
    evidence = _evidence(tmp_path / "ev", 0, {"0": patched["good_sha"]}, selection_seed=1)
    with pytest.raises(ValueError, match="disagrees on the bundle seed"):
        _run(patched, tmp_path / "out", evidence=evidence)
    assert patched["loaded"] == []


def test_val_a_reproduction_mismatch_stops_before_val_b(patched, tmp_path):
    # The checkpoint records a val-A EER that the weights do not give (recorded + 0.25, wrapped
    # into [0, 1]), so the reproduction check fails after val-A and val-B is never loaded.
    recorded = patched["checkpoint"]  # same weights, wrong recorded EER
    good = json.loads((patched["good"] / "checkpoint.json").read_text(encoding="utf-8"))
    wrong = math.fmod(good["val_a_eer"] + 0.25, 1.0)
    folder, sha = recorded("wrong_eer", val_a_eer=wrong)
    evidence = _evidence(tmp_path / "ev", 0, {"0": sha})
    with pytest.raises(ValueError, match="val-A reproduction failed"):
        _run(patched, tmp_path / "out", checkpoint=folder, evidence=evidence)
    assert patched["loaded"] == ["val-A"]
    assert not (tmp_path / "out").exists()


def test_incomplete_val_b_cache_is_refused(patched, tmp_path, monkeypatch):
    # F2 lists one more val-B row than the cache holds: every cached path is valid, but the cache
    # is incomplete, so the fit would silently use a subset of val-B.
    extra = {"path": "val-B/REAL/99.wav", "label": "REAL", "subgroup": "REAL", "group": "g99",
             "split": "val-B", "hash": "f" * 64}
    bigger = pd.concat([patched["manifest"], pd.DataFrame([extra])], ignore_index=True)
    monkeypatch.setattr(run_module, "load_f2_manifest", lambda p: bigger)
    with pytest.raises(ValueError, match="val-B: 1 F2 rows are missing from the cache"):
        _run(patched, tmp_path / "out")
    assert patched["loaded"] == ["val-A", "val-B"]
    assert not (tmp_path / "out").exists()


def test_existing_out_dir_is_refused(patched, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        _run(patched, out)
    assert patched["loaded"] == [] and list(out.iterdir()) == []


def test_failure_after_mkdir_removes_out_dir(patched, tmp_path, monkeypatch):
    def broken(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(run_module, "_write_npz", broken)
    with pytest.raises(OSError, match="disk full"):
        _run(patched, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_diagnostics_by_hand():
    # a = 1, b = 0, so p = sigmoid(s). exp(-800) underflows to 0 in float64, so s = -800 gives
    # p == 0.0 exactly and s = 800 gives p == 1.0 exactly (saturation).
    #   bonafide (REAL):    s = -800, -2, 0.5   -> p = 0, 0.119, 0.622
    #   spoof FAKE_mp3:     s = 3, 800          -> p = 0.953, 1
    #   spoof FAKE_other:   s = -1, 1           -> p = 0.269, 0.731
    labels = np.array([0, 0, 0, 1, 1, 1, 1])
    scores = np.array([-800.0, -2.0, 0.5, 3.0, 800.0, -1.0, 1.0])
    groups = np.array(["REAL", "REAL", "REAL", "FAKE_mp3", "FAKE_mp3", "FAKE_other", "FAKE_other"])
    policy = RiskPolicy(version="t", calibration=PlattCalibration(a=1.0, b=0.0, n_spoof=4, n_bonafide=3),
                        threshold=0.5, half_width=0.2, target_deferral=0.2, n_spoof_fit=4,
                        n_bonafide_fit=3, deferral_on_fit=1 / 7)
    before = (labels.copy(), scores.copy(), groups.copy())
    d = run_module.diagnostics(labels, scores, policy, groups)
    for a, b in zip(before, (labels, scores, groups)):
        np.testing.assert_array_equal(a, b)  # pure: inputs untouched
    json.dumps(d, allow_nan=False)  # strict JSON

    sig = lambda z: 1 / (1 + math.exp(-z))  # noqa: E731
    assert d["counts"] == {"n_bonafide": 3, "n_spoof": 4}
    # Gap: max bonafide raw 0.5 is above min spoof raw -1, so not separable.
    assert d["separation"] == {"max_bonafide_raw": 0.5, "min_spoof_raw": -1.0,
                               "max_bonafide_p": pytest.approx(sig(0.5)),
                               "min_spoof_p": pytest.approx(sig(-1)), "separable": False}
    # Saturation: one bonafide at p == 0, one spoof at p == 1.
    assert d["score_level"]["n_p_equal_0"] == {"bonafide": 1, "spoof": 0}
    assert d["score_level"]["n_p_equal_1"] == {"bonafide": 0, "spoof": 1}
    # AUC: spoof -1 beats 2 of 3 bonafide, the other 3 spoof beat all 3 -> 11/12. No ties in p,
    # so p gives the same value here.
    # EER: descending, (FPR, FNR) goes (0, .25) at s = 1, then (1/3, .25) at s = 0.5; the crossing
    # is interpolated with w = .25 / (.25 + 1/12) = .75 -> EER = .75 * 1/3 = 0.25.
    for level in ("raw", "p"):
        assert d["score_level"][level] == {"eer": pytest.approx(0.25), "roc_auc": pytest.approx(11 / 12)}
    assert d["policy"] == {"a": 1.0, "b": 0.0, "threshold": 0.5, "half_width": 0.2,
                           "target_deferral": 0.2, "deferral_on_fit": 1 / 7}
    # Band [0.3, 0.7): bonafide 0, 0.119 authentic, 0.622 uncertain; spoof 0.953, 1, 0.731
    # synthetic, 0.269 authentic.
    assert d["decisions"] == {"bonafide": {"authentic": 2, "uncertain": 1, "synthetic": 0},
                              "spoof": {"authentic": 1, "uncertain": 0, "synthetic": 3}}
    # Forced binary at 0.5: bonafide 0.622 -> synthetic (1 FP); spoof 0.269 -> authentic (1 FN).
    assert d["forced_binary"] == {"false_positives": 1, "false_negatives": 1}
    # REAL vs FAKE_mp3: spoof {3, 800} above every bonafide -> EER 0.
    # REAL vs FAKE_other: (FPR, FNR) = (0, .5) at s = 1, (1/3, .5) at 0.5, (1/3, 0) at -1;
    # w = (1/6) / (1/6 + 1/3) = 1/3 -> EER = 1/3 + 1/3 * 0 = 1/3.
    sub = d["subgroups"]
    assert sub["counts"] == {"REAL": 3, "FAKE_mp3": 2, "FAKE_other": 2}
    assert sub["REAL_vs_FAKE_mp3"] == {"eer": 0.0, "n_bonafide": 3, "n_spoof": 2}
    assert sub["REAL_vs_FAKE_other"] == {"eer": pytest.approx(1 / 3), "n_bonafide": 3, "n_spoof": 2}
    assert d["purpose"].startswith("descriptive only")


def test_module_has_no_train_test_external_path_or_wandb():
    tree = ast.parse(RUN_FILE.read_text(encoding="utf-8"))
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for forbidden in ("train", "test", "external"):
        assert forbidden not in strings
    imported = {alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any("wandb" in name for name in imported)
    assert (run_module.CHECK_SPLIT, run_module.FIT_SPLIT) == ("val-A", "val-B")


def test_cli_parses_required_arguments(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_run(*args, **kw):
        seen["args"], seen["kw"] = args, kw
        return {"ok": 1}

    monkeypatch.setattr(run_module, "run", fake_run)
    run_module.main(["--cache-dir", "c", "--checkpoint", "k", "--profile", "p", "--config", "g",
                     "--manifest", "m", "--stage8-evidence", "e", "--out", str(tmp_path / "o"),
                     "--version", "risk-v1"])
    assert [str(a) for a in seen["args"]] == ["c", "k", "p", "g", "m", "e", str(tmp_path / "o")]
    assert seen["kw"] == {"version": "risk-v1"}
    assert json.loads(capsys.readouterr().out) == {"ok": 1}
