"""Model v2 search (D49): selection, seeds, refusals, resumability, reports. Synthetic, scripted training."""
from __future__ import annotations

import ast
import functools
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests._torch import require_torch

torch = require_torch()

import src.tuning.v2_search as vs  # noqa: E402
from src.evaluation.seeds import select_bundle_seed  # noqa: E402
from src.models.v2_checkpoint import build_model, load_v2_checkpoint, trainable_parameters  # noqa: E402
from src.preprocessing.profile import PreprocessingError  # noqa: E402
from src.training.selection_set import load_selection_set  # noqa: E402
from src.training.train_v2 import V2TrainResult  # noqa: E402
from src.training.v2_config import load_v2_config  # noqa: E402
from tests._v2_fakes import (  # noqa: E402
    COMMIT,
    PROFILE_PATH,
    S_COUNTS,
    V2_CONFIG_PATH,
    FakeCodec,
    aug_config_copy,
    build_caches,
    fake_resources,
    profile as load_test_profile,
    selection_arrays,
    sha,
    v2_config_copy,
    write_dataset,
    write_s,
)

SEARCH_FILE = Path(vs.__file__)
PARAMS = {"cnn-aug": 97761, "crnn": 319585, "aasist": 297705}
ORDER = ("cnn-aug", "crnn", "aasist")


# --- selection rules ----------------------------------------------------------------------------

def test_select_candidate_lowest_m_then_parameters_then_order():
    assert vs.select_candidate({"cnn-aug": 0.2, "crnn": 0.1, "aasist": 0.3}, PARAMS, ORDER) == ("crnn", ["crnn"])
    assert vs.select_candidate({"cnn-aug": 0.2, "crnn": 0.1, "aasist": 0.1}, PARAMS, ORDER) == (
        "aasist", ["aasist", "crnn"])  # fewer trainable parameters first
    assert vs.select_candidate(dict.fromkeys(ORDER, 0.0), PARAMS, ORDER) == ("cnn-aug", ["cnn-aug", "aasist", "crnn"])
    same = dict.fromkeys(ORDER, 5)
    assert vs.select_candidate({"cnn-aug": 0.3, "crnn": 0.1, "aasist": 0.1}, same, ORDER) == (
        "crnn", ["crnn", "aasist"])  # equal parameters: candidate order


def _scripted(ms: dict):
    calls = []

    def run_one(candidate, seed):
        calls.append((candidate, seed))
        return {"candidate": candidate, "seed": seed, "best_selection_m": ms[(candidate, seed)],
                "trainable_parameters": PARAMS[candidate]}
    return run_one, calls


@pytest.mark.parametrize("seed_ms", [{0: 0.1, 1: 0.3, 2: 0.2}, {0: 0.2, 1: 0.2, 2: 0.1}, {0: 0.3, 1: 0.1, 2: 0.2}])
def test_only_winner_gets_seeds_1_and_2_and_bundle_seed_rule(seed_ms):
    config = load_v2_config(V2_CONFIG_PATH)
    ms = {("cnn-aug", 0): 0.5, ("crnn", 0): seed_ms[0], ("aasist", 0): 0.4,
          ("crnn", 1): seed_ms[1], ("crnn", 2): seed_ms[2]}
    run_one, calls = _scripted(ms)
    out = vs.search_and_select(config, run_one)
    assert calls == [("cnn-aug", 0), ("crnn", 0), ("aasist", 0), ("crnn", 1), ("crnn", 2)]
    assert out["winner"] == "crnn" and out["seeds"][0] is out["search"]["crnn"]
    assert out["bundle_seed"] == select_bundle_seed(seed_ms)


# --- end to end with scripted train_v2 -----------------------------------------------------------

@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    base = tmp_path_factory.mktemp("v2search")
    prof = load_test_profile()
    root, manifest, manifest_path = write_dataset(base)
    raw_dir, lm = build_caches(base, root, manifest, manifest_path, prof)
    aug_path = aug_config_copy(base / "augmentation.yaml")
    arrays = selection_arrays(root, manifest, prof)
    write_s(base / "S", arrays, prof, sha(manifest_path), sha(aug_path), recompute_rows=1)
    write_s(base / "S_other_aug", arrays, prof, sha(manifest_path), "d" * 64, recompute_rows=1)
    return {"base": base, "raw": raw_dir, "lm": lm, "manifest_path": manifest_path,
            "aug": aug_path, "aug_unpinned": aug_config_copy(base / "aug_unpinned.yaml", pinned=False),
            "v2": v2_config_copy(base / "v2.yaml")}


class PatchedCodec(FakeCodec):
    instances = []

    def __init__(self, settings, sample_rate):
        super().__init__()
        PatchedCodec.instances.append(self)


M_SCRIPT = {("cnn-aug", 0): 0.30, ("crnn", 0): 0.10, ("aasist", 0): 0.20, ("crnn", 1): 0.15,
            ("crnn", 2): 0.05}


@pytest.fixture()
def patched(inputs, monkeypatch):
    calls = []

    def fake_train_v2(candidate, *, config, run_seed, raw_cache, selection, augmenter, profile, mean,
                      std, max_epochs, patience, batch_size, num_workers, logger, device):
        calls.append({"candidate": candidate, "seed": run_seed, "logmel": mean is not None,
                      "form": selection.form, "batch_size": batch_size, "max_epochs": max_epochs})
        torch.manual_seed(run_seed)
        model = build_model(candidate)
        m = M_SCRIPT[(candidate, run_seed)]
        return V2TrainResult(candidate=candidate, state_dict=model.state_dict(), best_epoch=1,
                             best_selection_m=m, epochs_run=1,
                             history=({"epoch": 1, "selection_m": m},),
                             determinism={"seed": run_seed, "num_workers": num_workers},
                             peak_gpu_memory_bytes=None, trainable_parameters=trainable_parameters(model))

    monkeypatch.setattr(vs, "train_v2", fake_train_v2)
    monkeypatch.setattr(vs, "code_version", lambda: {"git_commit": COMMIT, "git_dirty": False})
    monkeypatch.setattr(vs, "load_f2_manifest", lambda p: pd.read_csv(p, dtype=str, keep_default_na=False))
    monkeypatch.setattr(vs, "F2_MANIFEST_SHA256", sha(inputs["manifest_path"]))
    monkeypatch.setattr(vs, "FFmpegCodecBackend", PatchedCodec)
    monkeypatch.setattr(vs, "load_selection_set", functools.partial(
        load_selection_set, expected_n_rows=10, expected_counts=S_COUNTS, recompute_rows=1))
    PatchedCodec.instances.clear()
    return calls


def _run(inputs, out, **changes):
    args = dict(raw_cache_dir=inputs["raw"], stage8_train_logmel_dir=inputs["lm"] / "train",
                selection_set_dir=inputs["base"] / "S", profile_path=PROFILE_PATH,
                v2_config_path=inputs["v2"], augmentation_config_path=inputs["aug"],
                manifest_path=inputs["manifest_path"], resources=fake_resources(), out_dir=out,
                version="cnn-v2-check", logger_mode="disabled", device="cpu", num_workers=0)
    args.update(changes)
    return vs.run_search(**args)


def test_end_to_end_outputs(inputs, patched, tmp_path, capsys):
    report = _run(inputs, tmp_path / "out")
    assert "removed 0 incomplete" in capsys.readouterr().out
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == ["run_report.json", "runs", "search_report.json",
                                                    "selection_report.json"]
    assert sorted(p.name for p in (out / "runs").iterdir()) == [
        "aasist_seed0", "cnn-aug_seed0", "crnn_seed0", "crnn_seed1", "crnn_seed2"]
    assert [(c["candidate"], c["seed"]) for c in patched] == [
        ("cnn-aug", 0), ("crnn", 0), ("aasist", 0), ("crnn", 1), ("crnn", 2)]
    assert [(c["logmel"], c["form"]) for c in patched[:3]] == [(True, "logmel"), (True, "logmel"),
                                                              (False, "waveform")]
    assert {c["batch_size"] for c in patched} == {8} and {c["max_epochs"] for c in patched} == {30}
    assert PatchedCodec.instances[0].verified == 1  # verify_pin once at start

    assert report == json.loads((out / "run_report.json").read_text(encoding="utf-8"))
    assert (report["selected"], report["bundle_seed"]) == ("crnn", select_bundle_seed({0: 0.1, 1: 0.15, 2: 0.05}))
    assert report["code_commit"] == COMMIT and report["augmentation_config_sha256"] == sha(inputs["aug"])
    assert report["selection_set_sha256"] == sha(inputs["base"] / "S" / "selection_set_manifest.json")
    stage8 = json.loads((inputs["lm"] / "train" / "train.json").read_text(encoding="utf-8"))["npz_sha256"]
    assert report["train_stats_npz_sha256"]["aasist_seed0"] is None
    assert report["train_stats_npz_sha256"]["crnn_seed1"] == stage8
    selection = json.loads((out / "selection_report.json").read_text(encoding="utf-8"))
    assert selection["selected"] == "crnn" and selection["tie_rule_decided"] is False
    assert [selection["seeds"][s]["reused_search_run"] for s in "012"] == [True, False, False]
    for name, digest in report["checkpoint_sha256"].items():
        model, mean, std, meta = load_v2_checkpoint(out / "runs" / name / "checkpoint", expected_sha256=digest)
        run = json.loads((out / "runs" / name / "run.json").read_text(encoding="utf-8"))
        assert meta["selection_m"] == run["best_selection_m"] == M_SCRIPT[(run["candidate"], run["seed"])]
        assert meta["selection_set_sha256"] == report["selection_set_sha256"]
        assert meta["determinism"]["num_workers"] == 0
        assert (mean is None) == (run["candidate"] == "aasist")
        assert run["provenance"]["train_stats_npz_sha256"] == meta["train_stats_npz_sha256"]


def test_tie_rule_recorded(inputs, patched, tmp_path, monkeypatch):
    monkeypatch.setitem(M_SCRIPT, ("cnn-aug", 0), 0.10)
    monkeypatch.setitem(M_SCRIPT, ("cnn-aug", 1), 0.2)
    monkeypatch.setitem(M_SCRIPT, ("cnn-aug", 2), 0.3)
    report = _run(inputs, tmp_path / "out")
    assert report["selected"] == "cnn-aug" and report["tie_rule_decided"] is True


@pytest.mark.parametrize("key, value, match", [
    ("v2_config_path", V2_CONFIG_PATH, "batch_size is null"),
    ("augmentation_config_path", "aug_unpinned", "pins must be complete"),
    ("selection_set_dir", "S_other_aug", "augmentation_config_sha256"),
    ("logger_mode", "online-ish", "logger_mode"),
    ("num_workers", -1, "num_workers"),
])
def test_refusals(inputs, patched, tmp_path, key, value, match):
    value = {"aug_unpinned": inputs["aug_unpinned"], "S_other_aug": inputs["base"] / "S_other_aug"}.get(value, value)
    with pytest.raises(ValueError, match=match):
        _run(inputs, tmp_path / "out", **{key: value})
    assert patched == []


def test_refuses_dirty_worktree_before_reading(inputs, patched, tmp_path, monkeypatch):
    def dirty():
        raise PreprocessingError("Official Stage 6 validation requires a clean Git worktree")
    monkeypatch.setattr(vs, "code_version", dirty)
    monkeypatch.setattr(vs, "load_v2_config", lambda p: pytest.fail("config read before the worktree check"))
    with pytest.raises(PreprocessingError, match="clean Git worktree"):
        _run(inputs, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_refuses_missing_resources_and_repo_output(inputs, patched, tmp_path):
    from src.augmentation import AugmentationResources
    with pytest.raises(ValueError, match="training halves"):
        _run(inputs, tmp_path / "out", resources=AugmentationResources())
    with pytest.raises(ValueError, match="inside the repository"):
        _run(inputs, PROFILE_PATH.parent / "v2_search_should_not_exist")


def test_refuses_stats_cache_mismatch(inputs, patched, tmp_path, monkeypatch):
    real = vs.load_logmel_cache

    def flipped(*args, **kw):
        features, labels, paths, meta = real(*args, **kw)
        return features, (1 - labels).astype(np.int8), paths, meta
    monkeypatch.setattr(vs, "load_logmel_cache", flipped)
    with pytest.raises(ValueError, match="differ from the Stage 8 train cache"):
        _run(inputs, tmp_path / "out")


def test_resume_removes_partial_reuses_complete_and_keeps_reports(inputs, patched, tmp_path, capsys):
    out = tmp_path / "out"
    first = _run(inputs, out)
    runs = out / "runs"
    shutil.rmtree(runs / "crnn_seed2")
    (runs / "crnn_seed2.partial").mkdir()
    (runs / "crnn_seed2.partial" / "junk").write_text("x")
    (out / "run_report.json").unlink()
    patched.clear()
    capsys.readouterr()
    again = _run(inputs, out)
    assert "removed 1 incomplete" in capsys.readouterr().out
    assert [(c["candidate"], c["seed"]) for c in patched] == [("crnn", 2)]  # only the missing run
    assert again["checkpoint_sha256"] == first["checkpoint_sha256"]
    patched.clear()
    assert _run(inputs, out) == again and patched == []  # complete: verified and returned


def test_resume_refuses_changed_provenance_and_reports_never_overwritten(inputs, patched, tmp_path):
    out = tmp_path / "out"
    _run(inputs, out)
    changed = v2_config_copy(tmp_path / "v2_changed.yaml", {"cnn-aug": 16, "crnn": 8, "aasist": 8})
    with pytest.raises(ValueError, match="provenance differs"):
        _run(inputs, out, v2_config_path=changed)
    path = out / "selection_report.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"crnn"', '"aasist"', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="never overwritten"):
        _run(inputs, out)
    run_json = out / "runs" / "aasist_seed0" / "run.json"
    data = json.loads(run_json.read_text(encoding="utf-8"))
    data["checkpoint_sha256"] = "0" * 64
    run_json.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        _run(inputs, out)


def test_module_has_no_val_b_test_or_external_path():
    tree = ast.parse(SEARCH_FILE.read_text(encoding="utf-8"))
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for forbidden in ("val-B", "val_b", "test", "external"):
        assert forbidden not in strings
    assert vs.TRAIN_SPLIT == "train"
    assert not hasattr(vs, "main")  # no CLI in this PR (wired in V2b-4)
