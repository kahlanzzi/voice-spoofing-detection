"""Model v2 risk-policy run (PR V4a): evidence binding, provenance, exact M reproduction, val-B fit.

Synthetic WAVs, F2-like manifest, a Stage 8-like val-B log-Mel cache, a 10-clip S (tests/_v2_fakes.py)
and seeded, untrained v2 checkpoints whose selection_m is computed by train_v2's own functions on
CPU (what train_v2 records at the end of an epoch). CPU; no W&B; no real data.
Convention: label 1 = spoof (FAKE), 0 = bonafide (REAL).
"""
from __future__ import annotations

import ast
import functools
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests._torch import require_torch

torch = require_torch()

import src.tuning.risk_run as stage10  # noqa: E402
import src.tuning.v2_risk_run as vr  # noqa: E402
from src.evaluation.metrics import roc_auc  # noqa: E402
from src.features.logmel_cache import build_logmel_cache, load_logmel_cache  # noqa: E402
from src.models.v2_checkpoint import (  # noqa: E402
    INPUT_SHAPES,
    INPUTS,
    build_model,
    save_v2_checkpoint,
    trainable_parameters,
    v2_checkpoint_sha256,
)
from src.preprocessing.validation import read_verified_audio  # noqa: E402
from src.preprocessing.waveform import preprocess_waveform  # noqa: E402
from src.risk.policy import RiskPolicy  # noqa: E402
from src.training.normalization import fit_bin_stats  # noqa: E402
from src.training.selection_set import CONDITIONS, load_selection_set  # noqa: E402
from src.training.train_v2 import evaluate_selection, selection_m, selection_tensors  # noqa: E402
from tests._v2_fakes import (  # noqa: E402
    COMMIT,
    PROFILE_PATH,
    S_COUNTS,
    profile as load_test_profile,
    selection_arrays,
    sha,
    v2_config_copy,
    write_dataset,
    write_s,
)

RUN_FILE = Path(vr.__file__)
SIZES = {"val-A": 10, "val-B": 12}  # val-A feeds S only; val-B is the fit split
OUTPUTS = sorted(["diagnostics_report.json", "risk_policy.json", "run_report.json",
                  "val_b_scores.npz"])
load_s = functools.partial(load_selection_set, expected_n_rows=10, expected_counts=S_COUNTS,
                           recompute_rows=1)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    base = tmp_path_factory.mktemp("v2risk")
    prof = load_test_profile()
    root, manifest, manifest_path = write_dataset(base, sizes=SIZES)
    # Both D14 FAKE subgroups on val-B, so the Stage 10 diagnostics have every subgroup.
    fake_b = manifest.index[(manifest.split == "val-B") & (manifest.label == "FAKE")]
    manifest.loc[fake_b[::2], "subgroup"] = "FAKE_mp3"
    manifest.to_csv(manifest_path, index=False)
    f2 = sha(manifest_path)
    cache_dir = base / "logmel"
    build_logmel_cache(root, manifest, prof, "val-B", cache_dir / "val-B", f2_manifest_sha256=f2,
                       profile_config_sha256=sha(PROFILE_PATH), code_commit=COMMIT)
    s_dir = base / "S"
    s_sha = write_s(s_dir, selection_arrays(root, manifest, prof), prof, f2, "a" * 64,
                    recompute_rows=1)
    v2 = v2_config_copy(base / "v2.yaml")  # batch size 8 each
    s_forms = {form: load_s(s_dir, prof, form=form) for form in ("logmel", "waveform")}
    mean, std = fit_bin_stats(s_forms["logmel"].features("clean"))

    def waveform_source():
        rows = manifest[manifest.split == "val-B"].sort_values("path", kind="stable")
        waves = np.stack([preprocess_waveform(read_verified_audio(root, row)[0], 16000, prof).waveform
                          for _, row in rows.iterrows()]).astype(np.float32)
        labels = rows["label"].map({"FAKE": 1, "REAL": 0}).to_numpy(dtype=np.int8)
        return waves, labels, np.array(rows["path"].tolist())

    val_b_logmel = load_logmel_cache(cache_dir / "val-B", "val-B", prof, expected_f2_manifest_sha256=f2)
    val_b = {"logmel": val_b_logmel[0], "waveform": waveform_source()[0]}
    val_b_labels = val_b_logmel[1]

    def oriented_model(arch, seed):
        # A seeded, untrained model whose val-B scores rank spoof above bonafide: if they rank
        # backwards (Platt would refuse a <= 0), the one-logit output layer is negated. Test fixture
        # only; deterministic on CPU.
        torch.manual_seed(seed)
        model = build_model(arch).eval()
        kind = INPUTS[arch]
        m_mean, m_std = (mean, std) if kind == "logmel" else (None, None)
        x = vr.model_inputs(kind, val_b[kind], m_mean, m_std)
        if roc_auc(val_b_labels, vr.score_clips(model, x, 8)) < 0.5:
            out = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
            assert out.out_features == 1
            with torch.no_grad():
                out.weight.neg_()
                out.bias.neg_()
        return model

    def checkpoint(arch, seed, name=None, **changes):
        model = oriented_model(arch, seed)
        kind = INPUTS[arch]
        m_mean, m_std = (mean, std) if kind == "logmel" else (None, None)
        tensors = selection_tensors(s_forms[kind], kind, m_mean, m_std)
        eers, _ = evaluate_selection(model, tensors, s_forms[kind].labels, 8, torch.device("cpu"))
        meta = {
            "schema_version": 1, "architecture": arch, "version": "cnn-v2-test", "input": kind,
            "input_shape": list(INPUT_SHAPES[kind]), "seed": seed, "best_epoch": 1,
            "selection_m": selection_m(eers), "code_commit": COMMIT,
            "environment": {"python": "3.13.15"}, "profile_fingerprint": prof.fingerprint(),
            "f2_manifest_sha256": f2, "augmentation_config_sha256": "a" * 64,
            "selection_set_sha256": s_sha,
            "train_stats_npz_sha256": "b" * 64 if kind == "logmel" else None,
            "determinism": {"seed": seed}, "trainable_parameters": trainable_parameters(model),
        }
        meta.update(changes)
        folder = save_v2_checkpoint(model, meta, base / "ckpt" / (name or f"{arch}_{seed}"),
                                    mean=m_mean, std=m_std)
        return folder, v2_checkpoint_sha256(folder)

    good = {arch: checkpoint(arch, 0) for arch in ("cnn-aug", "crnn", "aasist")}
    return {"base": base, "manifest": manifest, "manifest_path": manifest_path,
            "cache_dir": cache_dir, "s_dir": s_dir, "s_sha": s_sha, "v2": v2,
            "checkpoint": checkpoint, "good": good, "waveform_source": waveform_source}


def _evidence(folder: Path, winner: str, seed: int, hashes: dict, *, selection_winner=None,
              selection_seed=None) -> Path:
    """A minimal v2 search evidence folder (run_report + selection_report)."""
    folder.mkdir(parents=True)
    (folder / "run_report.json").write_text(json.dumps(
        {"selected": winner, "bundle_seed": seed, "checkpoint_sha256": hashes}), encoding="utf-8")
    (folder / "selection_report.json").write_text(json.dumps(
        {"selected": winner if selection_winner is None else selection_winner,
         "bundle_seed": seed if selection_seed is None else selection_seed}), encoding="utf-8")
    return folder


@pytest.fixture()
def patched(inputs, monkeypatch, tmp_path):
    """Clean worktree, the synthetic manifest as F2, a 10-clip S loader, load spies."""
    monkeypatch.setattr(vr, "code_version", lambda: {"git_commit": COMMIT, "git_dirty": False})
    monkeypatch.setattr(vr, "load_f2_manifest",
                        lambda p: pd.read_csv(p, dtype=str, keep_default_na=False))
    monkeypatch.setattr(stage10, "F2_MANIFEST_SHA256", sha(inputs["manifest_path"]))
    loaded = []
    real_cache = stage10.load_logmel_cache

    def cache_spy(out_dir, split, *args, **kw):
        loaded.append(split)
        return real_cache(out_dir, split, *args, **kw)

    def s_spy(folder, profile, *, form, **kw):
        loaded.append(f"S:{form}")
        return load_s(folder, profile, form=form, **kw)

    monkeypatch.setattr(stage10, "load_logmel_cache", cache_spy)
    monkeypatch.setattr(vr, "load_selection_set", s_spy)

    def use_evidence(winner, seed, hashes, **kw):
        folder = _evidence(tmp_path / f"ev_{len(list(tmp_path.glob('ev_*')))}", winner, seed,
                           hashes, **kw)
        monkeypatch.setattr(vr, "V2_SEARCH_EVIDENCE", folder)
        return folder

    return {**inputs, "loaded": loaded, "use_evidence": use_evidence}


def _good_evidence(patched, arch):
    folder, digest = patched["good"][arch]
    patched["use_evidence"](arch, 0, {f"{arch}_seed0": digest, f"{arch}_seed1": "1" * 64})
    return folder


def _run(patched, out, checkpoint, **kw):
    if "cache_dir" not in kw and "val_b_waveforms" not in kw:
        kw["cache_dir"] = patched["cache_dir"]
    return vr.run(checkpoint, patched["s_dir"], PROFILE_PATH, patched["v2"],
                  patched["manifest_path"], out, version="risk-test", **kw)


def _check_outputs(patched, out, report, arch, val_b_input):
    assert sorted(p.name for p in out.iterdir()) == OUTPUTS
    data = json.loads((out / "run_report.json").read_text(encoding="utf-8"))
    assert data == report and list(data) == sorted(data)
    assert set(data) == {
        "schema_version", "version", "code_commit", "environment", "profile_fingerprint",
        "profile_config_sha256", "f2_manifest_sha256", "provenance_checks", "v2_search_evidence",
        "architecture", "bundle_seed", "checkpoint_sha256", "val_b_input", "scoring",
        "selection_m_recomputed", "selection_eers_recomputed", "risk_policy_sha256",
        "val_b_scores_sha256", "headline",
    }
    ckpt_dir, digest = patched["good"][arch]
    ckpt = json.loads((ckpt_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert (data["schema_version"], data["version"], data["code_commit"]) == (1, "risk-test", COMMIT)
    assert (data["architecture"], data["bundle_seed"], data["checkpoint_sha256"]) == (arch, 0, digest)
    assert data["scoring"] == {"device": "cpu", "batch_size": 8}  # from the v2 config copy
    assert data["selection_m_recomputed"] == ckpt["selection_m"]  # exact
    assert list(data["selection_eers_recomputed"]) == sorted(CONDITIONS)  # sorted keys in JSON
    assert math.fsum(data["selection_eers_recomputed"].values()) / 5 == ckpt["selection_m"]
    f2 = sha(patched["manifest_path"])
    assert data["provenance_checks"] == {
        "profile_fingerprint": {"expected": ckpt["profile_fingerprint"],
                                "actual": ckpt["profile_fingerprint"]},
        "f2_manifest_sha256": {"expected": f2, "actual": f2},
        "selection_set_sha256": {"expected": patched["s_sha"], "actual": patched["s_sha"]},
    }
    ev = vr.V2_SEARCH_EVIDENCE
    assert data["v2_search_evidence"] == {n: sha(ev / n) for n in ("run_report.json",
                                                                   "selection_report.json")}
    assert data["val_b_input"] == val_b_input
    expected = RiskPolicy.from_json((out / "risk_policy.json").read_text(encoding="utf-8")).to_json()
    assert (out / "risk_policy.json").read_bytes() == expected.encode("utf-8")
    assert data["risk_policy_sha256"] == hashlib.sha256(expected.encode("utf-8")).hexdigest()
    assert data["val_b_scores_sha256"] == sha(out / "val_b_scores.npz")
    for name in ("risk_policy.json", "diagnostics_report.json", "run_report.json"):
        assert b"\r\n" not in (out / name).read_bytes(), name
    assert (data["headline"]["n_bonafide"], data["headline"]["n_spoof"]) == (6, 6)
    with np.load(io.BytesIO((out / "val_b_scores.npz").read_bytes()), allow_pickle=False) as npz:
        assert sorted(npz.files) == ["labels", "p", "paths", "score"]
        manifest = patched["manifest"]
        assert npz["paths"].tolist() == sorted(manifest.loc[manifest.split == "val-B", "path"])
        assert npz["score"].dtype == np.float64 and npz["p"].dtype == np.float64
        policy = RiskPolicy.from_json(expected)
        np.testing.assert_array_equal(npz["p"], policy.calibration.transform(npz["score"]))
    diag = json.loads((out / "diagnostics_report.json").read_text(encoding="utf-8"))
    assert diag["purpose"].startswith("descriptive only")
    assert diag["subgroups"]["counts"] == {"REAL": 6, "FAKE_mp3": 3, "FAKE_other": 3}


@pytest.mark.parametrize("arch", ["crnn", "cnn-aug"])
def test_logmel_round_trip_deterministic_and_reads_s_then_val_b_only(patched, tmp_path, arch):
    checkpoint = _good_evidence(patched, arch)
    out = tmp_path / "run"
    report = _run(patched, out, checkpoint)
    npz = patched["cache_dir"] / "val-B" / "val-B.npz"
    _check_outputs(patched, out, report, arch,
                   {"form": "logmel", "npz_sha256": sha(npz), "n_rows": 12})
    assert patched["loaded"] == ["S:logmel", "val-B"]  # no other split (D9, D31)
    again = tmp_path / "again"
    _run(patched, again, checkpoint)
    for name in OUTPUTS:
        assert (again / name).read_bytes() == (out / name).read_bytes(), name


def test_aasist_round_trip_with_waveform_source(patched, tmp_path):
    checkpoint = _good_evidence(patched, "aasist")
    calls = []

    def source():
        calls.append(1)
        return patched["waveform_source"]()

    out = tmp_path / "run"
    report = _run(patched, out, checkpoint, val_b_waveforms=source)
    waves, labels, paths = patched["waveform_source"]()
    _check_outputs(patched, out, report, "aasist",
                   {"form": "waveform", "n_rows": 12,
                    "content_sha256": vr._sha256_arrays(waves, labels, paths)})
    assert patched["loaded"] == ["S:waveform"] and calls == [1]  # no log-Mel cache read
    again = tmp_path / "again"
    _run(patched, again, checkpoint, val_b_waveforms=source)
    for name in OUTPUTS:
        assert (again / name).read_bytes() == (out / name).read_bytes(), name


def test_aasist_without_waveform_source_is_refused(patched, tmp_path):
    checkpoint = _good_evidence(patched, "aasist")
    with pytest.raises(ValueError, match="F3 waveform val-B source is required and none exists yet"):
        _run(patched, tmp_path / "out", checkpoint)  # the CLI path: only a log-Mel cache_dir
    with pytest.raises(ValueError, match="F3 waveform val-B source is required"):
        vr.run(checkpoint, patched["s_dir"], PROFILE_PATH, patched["v2"], patched["manifest_path"],
               tmp_path / "out", version="risk-test")
    assert patched["loaded"] == [] and not (tmp_path / "out").exists()


def test_input_kind_mismatches_are_refused(patched, tmp_path):
    checkpoint = _good_evidence(patched, "crnn")
    with pytest.raises(ValueError, match="cache_dir .* is required"):
        vr.run(checkpoint, patched["s_dir"], PROFILE_PATH, patched["v2"], patched["manifest_path"],
               tmp_path / "out", version="risk-test")
    with pytest.raises(ValueError, match="waveform val-B source must not be given"):
        _run(patched, tmp_path / "out", checkpoint, cache_dir=patched["cache_dir"],
             val_b_waveforms=patched["waveform_source"])
    aasist = _good_evidence(patched, "aasist")
    with pytest.raises(ValueError, match="cache_dir .* must not be given"):
        _run(patched, tmp_path / "out", aasist, cache_dir=patched["cache_dir"],
             val_b_waveforms=patched["waveform_source"])
    assert patched["loaded"] == []


@pytest.mark.parametrize("bad, match", [
    ({"winner": "crnn", "seed": 0, "selection_winner": "cnn-aug"}, "disagrees on the winner"),
    ({"winner": "crnn", "seed": 0, "selection_seed": 1}, "disagrees on the bundle seed"),
    ({"winner": "crnn", "seed": 1}, "no checkpoint_sha256 for crnn_seed1"),
    ({"winner": "lstm", "seed": 0}, "selected must be one of"),
    ({"winner": "crnn", "seed": True}, "bundle_seed must be an int"),
])
def test_evidence_inconsistency_is_refused(patched, tmp_path, bad, match):
    kw = {k: v for k, v in bad.items() if k.startswith("selection_")}
    patched["use_evidence"](bad["winner"], bad["seed"], {"crnn_seed0": patched["good"]["crnn"][1]}, **kw)
    with pytest.raises(ValueError, match=match):
        _run(patched, tmp_path / "out", patched["good"]["crnn"][0])
    assert patched["loaded"] == [] and not (tmp_path / "out").exists()


def test_missing_evidence_folder_is_refused(patched, tmp_path, monkeypatch):
    monkeypatch.setattr(vr, "V2_SEARCH_EVIDENCE", tmp_path / "nowhere")
    with pytest.raises(ValueError, match="v2 search evidence .* is missing"):
        _run(patched, tmp_path / "out", patched["good"]["crnn"][0])


def test_committed_evidence_path_is_fixed():
    assert vr.V2_SEARCH_EVIDENCE == Path(__file__).resolve().parents[2] / "docs" / "audit_evidence" / "v2_search"


def test_checkpoint_hash_mismatch_is_refused_before_any_load(patched, tmp_path):
    patched["use_evidence"]("crnn", 0, {"crnn_seed0": "0" * 64})
    with pytest.raises(ValueError, match="checkpoint_sha256 mismatch"):
        _run(patched, tmp_path / "out", patched["good"]["crnn"][0])
    assert patched["loaded"] == []


def test_wrong_architecture_is_refused(patched, tmp_path):
    # The cnn-aug checkpoint's own hash listed as the crnn winner's: the hash passes, the
    # architecture does not (D45: the winner's bundle seed only).
    folder, digest = patched["good"]["cnn-aug"]
    patched["use_evidence"]("crnn", 0, {"crnn_seed0": digest})
    with pytest.raises(ValueError, match="architecture 'cnn-aug' is not the v2 search winner 'crnn'"):
        _run(patched, tmp_path / "out", folder)
    assert patched["loaded"] == []


def test_wrong_seed_is_refused(patched, tmp_path):
    folder, digest = patched["checkpoint"]("crnn", 1)
    patched["use_evidence"]("crnn", 0, {"crnn_seed0": digest})
    with pytest.raises(ValueError, match="seed 1 is not the v2 search bundle seed 0"):
        _run(patched, tmp_path / "out", folder)
    assert patched["loaded"] == []


@pytest.mark.parametrize("key, loaded", [
    ("profile_fingerprint", []),
    ("f2_manifest_sha256", []),
    ("selection_set_sha256", ["S:logmel"]),  # known only once S is loaded; before val-B
])
def test_provenance_mismatch_is_refused_before_val_b(patched, tmp_path, key, loaded):
    folder, digest = patched["checkpoint"]("crnn", 0, name=f"wrong_{key}", **{key: "0" * 64})
    patched["use_evidence"]("crnn", 0, {"crnn_seed0": digest})
    with pytest.raises(ValueError, match=f"provenance check failed: {key}"):
        _run(patched, tmp_path / "out", folder)
    assert patched["loaded"] == loaded
    assert not (tmp_path / "out").exists()


def test_m_mismatch_is_refused_with_both_values_and_eers(patched, tmp_path):
    good = json.loads((patched["good"]["crnn"][0] / "checkpoint.json").read_text(encoding="utf-8"))
    wrong = math.fmod(good["selection_m"] + 0.25, 1.0)
    folder, digest = patched["checkpoint"]("crnn", 0, name="wrong_m", selection_m=wrong)
    patched["use_evidence"]("crnn", 0, {"crnn_seed0": digest})
    with pytest.raises(ValueError, match="S reproduction failed") as info:
        _run(patched, tmp_path / "out", folder)
    message = str(info.value)
    assert repr(good["selection_m"]) in message and repr(wrong) in message
    for condition in CONDITIONS:
        assert condition in message
    assert patched["loaded"] == ["S:logmel"]  # val-B never read
    assert not (tmp_path / "out").exists()


def test_m_mismatch_by_one_ulp_is_refused(patched, tmp_path):
    good = json.loads((patched["good"]["crnn"][0] / "checkpoint.json").read_text(encoding="utf-8"))
    nudged = math.nextafter(good["selection_m"], 1.0)
    folder, digest = patched["checkpoint"]("crnn", 0, name="ulp_m", selection_m=nudged)
    patched["use_evidence"]("crnn", 0, {"crnn_seed0": digest})
    with pytest.raises(ValueError, match="S reproduction failed"):
        _run(patched, tmp_path / "out", folder)


@pytest.mark.parametrize("change, match", [
    (lambda w, l, p: (w[:, :100], l, p), "waveforms must be float32"),
    (lambda w, l, p: (w, l.astype(np.int64), p), "labels must be a non-empty int8"),
    (lambda w, l, p: (w, l, p[::-1].copy()), "sorted and unique"),
    (lambda w, l, p: (w[1:], l[1:], p[1:]), "1 F2 rows are missing"),
])
def test_waveform_source_is_checked(patched, tmp_path, change, match):
    checkpoint = _good_evidence(patched, "aasist")
    with pytest.raises(ValueError, match=match):
        _run(patched, tmp_path / "out", checkpoint,
             val_b_waveforms=lambda: change(*patched["waveform_source"]()))
    assert not (tmp_path / "out").exists()


def test_existing_out_dir_is_refused(patched, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        _run(patched, out, _good_evidence(patched, "crnn"))
    assert patched["loaded"] == [] and list(out.iterdir()) == []


def test_failure_after_mkdir_removes_out_dir(patched, tmp_path, monkeypatch):
    def broken(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(vr, "_write_npz", broken)
    with pytest.raises(OSError, match="disk full"):
        _run(patched, tmp_path / "out", _good_evidence(patched, "crnn"))
    assert not (tmp_path / "out").exists()


def test_module_has_no_other_split_path_or_wandb():
    tree = ast.parse(RUN_FILE.read_text(encoding="utf-8"))
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for forbidden in ("train", "test", "external", "val-A"):
        assert forbidden not in strings
    imported = {alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any("wandb" in name for name in imported)
    assert vr.FIT_SPLIT == "val-B"
    args = {a.arg for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run"
            for a in (*n.args.args, *n.args.kwonlyargs)}
    assert not any("split" in a for a in args)


def test_cli_parses_required_arguments(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_run(*args, **kw):
        seen["args"], seen["kw"] = args, kw
        return {"ok": 1}

    monkeypatch.setattr(vr, "run", fake_run)
    vr.main(["--checkpoint", "k", "--selection-set", "s", "--profile", "p", "--config", "g",
             "--manifest", "m", "--cache-dir", "c", "--out", str(tmp_path / "o"),
             "--version", "risk-v2"])
    assert [str(a) for a in seen["args"]] == ["k", "s", "p", "g", "m", str(tmp_path / "o")]
    assert seen["kw"]["version"] == "risk-v2" and str(seen["kw"]["cache_dir"]) == "c"
    assert set(seen["kw"]) == {"version", "cache_dir"}
    assert json.loads(capsys.readouterr().out) == {"ok": 1}
