"""V2b-4 rehearsal (src/tuning/v2_rehearsal.py): report, clean check, determinism, forced resume, folder
rules and the version guard first. Synthetic inputs; real train_v2 on a 4-clip / 2-row subset, CPU."""
from __future__ import annotations

import ast
import functools
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from tests._torch import require_torch

torch = require_torch()

import src.tuning.v2_rehearsal as vr  # noqa: E402
import src.tuning.v2_search as vs  # noqa: E402
from src.augmentation.version_guard import required_versions  # noqa: E402
from src.features.raw_train_cache import load_raw_train_cache  # noqa: E402
from src.preprocessing.profile import PreprocessingError  # noqa: E402
from src.training.selection_set import load_selection_set  # noqa: E402
from tests._v2_fakes import (  # noqa: E402
    COMMIT,
    PROFILE_PATH,
    S_COUNTS,
    FakeCodec,
    SubsetRawCache,
    SubsetSelection,
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

NOW = datetime(2026, 11, 21, 22, 59, tzinfo=timezone.utc)  # 144 h before the deadline
PINS = required_versions()


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    base = tmp_path_factory.mktemp("v2rehearsal")
    prof = load_test_profile()
    root, manifest, manifest_path = write_dataset(base, sizes={"train": 40, "val-A": 10})
    raw_dir, lm = build_caches(base, root, manifest, manifest_path, prof)
    aug_path = aug_config_copy(base / "augmentation.yaml")
    write_s(base / "S", selection_arrays(root, manifest, prof), prof, sha(manifest_path), sha(aug_path),
            recompute_rows=1)
    evidence = base / "evidence" / "train_raw.json"
    evidence.parent.mkdir()
    shutil.copy2(raw_dir / "train_raw.json", evidence)
    return {"base": base, "raw": raw_dir, "lm": lm, "manifest_path": manifest_path, "aug": aug_path,
            "v2": v2_config_copy(base / "v2.yaml"), "evidence": evidence}


class PatchedCodec(FakeCodec):
    def __init__(self, settings, sample_rate):
        super().__init__()


@pytest.fixture()
def env(inputs, monkeypatch):
    """Patches for the synthetic inputs; records the order of the first steps and the train calls."""
    calls = []
    real_train = vr.train_v2

    def small(candidate, *, raw_cache, selection, batch_size, **kw):
        calls.append(("train", candidate, kw["run_seed"], kw["max_epochs"], batch_size,
                      kw.get("resume_dir") is not None))
        return real_train(candidate, raw_cache=SubsetRawCache(raw_cache, 4),
                          selection=SubsetSelection(selection, 1), batch_size=2, **kw)

    def resources(directory):
        calls.append(("resources", Path(directory)))
        return fake_resources(), {"npz_sha256": "a" * 64}

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(vr, "train_v2", small)
    monkeypatch.setattr(vr, "code_version", lambda: calls.append("code_version") or {"git_commit": COMMIT})
    monkeypatch.setattr(vr, "load_f2_manifest", lambda p: pd.read_csv(p, dtype=str, keep_default_na=False))
    monkeypatch.setattr(vr, "F2_MANIFEST_SHA256", sha(inputs["manifest_path"]))
    monkeypatch.setattr(vr, "FFmpegCodecBackend", PatchedCodec)
    monkeypatch.setattr(vr, "load_selection_set", functools.partial(
        load_selection_set, expected_n_rows=10, expected_counts=S_COUNTS, recompute_rows=1))
    monkeypatch.setattr(vs, "RAW_CACHE_EVIDENCE", inputs["evidence"])
    monkeypatch.setattr(vs, "load_v2_resources", resources)
    return calls


def _version(name: str) -> str:
    return PINS[name]


def _rehearse(inputs, out, **changes):
    args = dict(raw_cache_dir=inputs["raw"], stage8_train_logmel_dir=inputs["lm"] / "train",
                stage8_val_a_logmel_dir=inputs["lm"] / "val-A", selection_set_dir=inputs["base"] / "S",
                resources_dir=inputs["base"] / "resources", out_dir=out, device="cpu", num_workers=0,
                profile_path=PROFILE_PATH, v2_config_path=inputs["v2"],
                augmentation_config_path=inputs["aug"], manifest_path=inputs["manifest_path"],
                version=_version, now=lambda: NOW)
    args.update(changes)
    return vr.rehearse(**args)


@pytest.fixture(scope="module")
def done(inputs, tmp_path_factory):
    """One full rehearsal (module scope: the patches are applied by hand)."""
    mp = pytest.MonkeyPatch()
    calls = []
    try:
        real_train = vr.train_v2

        def small(candidate, *, raw_cache, selection, batch_size, **kw):
            calls.append((candidate, kw["run_seed"], kw["max_epochs"], kw.get("resume_dir") is not None))
            return real_train(candidate, raw_cache=SubsetRawCache(raw_cache, 4),
                              selection=SubsetSelection(selection, 1), batch_size=2, **kw)
        mp.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        mp.setattr(vr, "train_v2", small)
        mp.setattr(vr, "code_version", lambda: {"git_commit": COMMIT})
        mp.setattr(vr, "load_f2_manifest", lambda p: pd.read_csv(p, dtype=str, keep_default_na=False))
        mp.setattr(vr, "F2_MANIFEST_SHA256", sha(inputs["manifest_path"]))
        mp.setattr(vr, "FFmpegCodecBackend", PatchedCodec)
        mp.setattr(vr, "load_selection_set", functools.partial(
            load_selection_set, expected_n_rows=10, expected_counts=S_COUNTS, recompute_rows=1))
        mp.setattr(vs, "RAW_CACHE_EVIDENCE", inputs["evidence"])
        mp.setattr(vs, "load_v2_resources", lambda d: (fake_resources(), {"npz_sha256": "a" * 64}))
        out = tmp_path_factory.mktemp("rehearsal_out") / "stamp"
        report = _rehearse(inputs, out)
    finally:
        mp.undo()
    return {"report": report, "out": out, "calls": calls}


def test_report_written_once_with_every_section(done, inputs):
    report, out = done["report"], done["out"]
    assert json.loads((out / vs.REHEARSAL_REPORT).read_text(encoding="utf-8")) == report
    assert set(report) == {"schema_version", "kind", "require_versions", "environment", "device",
                           "num_workers", "batch_sizes", "inputs", "n_train", "clean_transform",
                           "epoch_timing", "determinism", "forced_resume", "cut_line"}
    assert report["kind"] == "v2_rehearsal" and report["require_versions"] == PINS
    assert report["inputs"]["code_commit"] == COMMIT
    assert report["inputs"]["raw_train_cache_npz_sha256"] == sha(inputs["raw"] / "train_raw.npz")
    assert report["n_train"] == 40 and report["device"] == "cpu" and report["num_workers"] == 0
    assert sorted(p.name for p in out.iterdir()) == sorted([vs.REHEARSAL_MARKER, vs.REHEARSAL_REPORT,
                                                           vr.FORCED_RESUME_DIR])
    marker = json.loads((out / vs.REHEARSAL_MARKER).read_text(encoding="utf-8"))
    assert marker["kind"] == "v2_rehearsal" and marker["code_commit"] == COMMIT


def test_clean_check_on_32_seeded_indices_of_both_classes(done, inputs):
    clean = done["report"]["clean_transform"]
    assert clean["passed"] is True and clean["n_checked"] == 32 == len(clean["indices"])
    assert clean["n_real"] == clean["n_fake"] == 16 and clean["n_rows"] == 40
    raw = load_raw_train_cache(inputs["raw"], expected_f2_manifest_sha256=sha(inputs["manifest_path"]))
    assert clean["indices"] == vr.clean_check_indices(raw.labels)  # fixed seed: same draw


def test_clean_check_indices_fixed_seeded_and_both_classes():
    import numpy as np
    labels = np.array([0, 1] * 50, dtype=np.int8)
    first = vr.clean_check_indices(labels)
    assert first == vr.clean_check_indices(labels) == sorted(first) and len(set(first)) == 32
    assert (labels[first] == 0).sum() == (labels[first] == 1).sum() == 16
    with pytest.raises(ValueError, match="needs 16 clips of label 1"):
        vr.clean_check_indices(np.zeros(40, dtype=np.int8))


def test_epoch_timing_per_candidate_with_the_configured_batch_sizes(done):
    report = done["report"]
    assert report["batch_sizes"] == {"cnn-aug": 8, "crnn": 8, "aasist": 8}  # the config copy
    timing = report["epoch_timing"]
    assert list(timing) == ["cnn-aug", "crnn", "aasist"]
    for row in timing.values():
        assert row["epoch_seconds"] > 0 and row["items_per_second"] > 0 and row["batch_size"] == 8
    assert done["calls"][:3] == [("cnn-aug", 0, 1, False), ("crnn", 0, 1, False), ("aasist", 0, 1, False)]


def test_determinism_two_runs_and_forced_resume_equal_on_cpu(done):
    report = done["report"]
    det, forced = report["determinism"], report["forced_resume"]
    assert done["calls"][3:] == [("crnn", 0, 2, False)] * 2 + [("crnn", 0, 2, True)] * 2
    assert (det["candidate"], det["seed"], det["epochs"]) == ("crnn", 0, 2)
    assert det["best_selection_m_equal"] is True and det["weights_sha256_equal"] is True
    assert det["history_equal_without_seconds"] is True
    assert det["run_1"]["weights_sha256"] == det["run_2"]["weights_sha256"]
    assert forced["state_last_epoch_at_crash"] == 1 and forced["epochs_trained_after_resume"] == [2]
    assert forced["best_selection_m_equal_to_run_1"] is True
    assert forced["weights_sha256_equal_to_run_1"] is True
    assert forced["history_equal_to_run_1_without_seconds"] is True
    assert "not promised on GPU" in det["note"]


def test_cut_line_estimate():
    line = vr.cut_line({"cnn-aug": 60.0, "crnn": 120.0, "aasist": 600.0}, 30,
                       ("cnn-aug", "crnn", "aasist"), NOW)
    assert line["worst_case_hours"] == pytest.approx((60 + 120 + 600 + 2 * 600) * 30 / 3600)  # 16.5
    assert line["worst_case_hours_without_aasist"] == pytest.approx((60 + 120 + 240) * 30 / 3600)
    assert line["hours_until_deadline"] == pytest.approx(144.0)
    assert line["fits"] is True and line["deadline_utc"] == "2026-11-27T22:59:00+00:00"
    late = vr.cut_line({"cnn-aug": 600.0, "crnn": 900.0, "aasist": 1200.0}, 30,
                       ("cnn-aug", "crnn", "aasist"), datetime(2026, 11, 26, 22, 59, tzinfo=timezone.utc))
    assert late["fits"] is False and late["fits_without_aasist"] is False and "drop aasist first" in late["rule"]


def test_require_versions_is_called_first(env, inputs, tmp_path):
    def wrong(name):
        return "0.0.0"
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="D64 augmentation library version mismatch"):
        _rehearse(inputs, out, version=wrong)
    assert env == [] and not out.exists()  # nothing else ran, nothing written


def test_dirty_worktree_refused_before_any_output(env, inputs, tmp_path, monkeypatch):
    def dirty():
        raise PreprocessingError("Official Stage 6 validation requires a clean Git worktree")
    monkeypatch.setattr(vr, "code_version", dirty)
    with pytest.raises(PreprocessingError, match="clean Git worktree"):
        _rehearse(inputs, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("planted", ["runs", "search_report.json", "selection_report.json", "run_report.json"])
def test_refuses_a_search_folder_as_output(env, inputs, tmp_path, planted):
    out = tmp_path / "search"
    out.mkdir()
    (out / planted).mkdir() if planted == "runs" else (out / planted).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="is a search folder"):
        _rehearse(inputs, out)
    assert sorted(p.name for p in out.iterdir()) == [planted]
    assert env == ["code_version"]


def test_refuses_a_non_empty_folder_and_a_repo_folder(env, inputs, tmp_path):
    out = tmp_path / "used"
    out.mkdir()
    (out / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        _rehearse(inputs, out)
    with pytest.raises(ValueError, match="inside the repository"):
        _rehearse(inputs, PROFILE_PATH.parent / "rehearsal_should_not_exist")


def test_marker_written_before_the_inputs_are_read(env, inputs, tmp_path, monkeypatch):
    out = tmp_path / "out"

    def pin(directory):
        assert (out / vs.REHEARSAL_MARKER).is_file()
        raise ValueError("stop after the marker")
    monkeypatch.setattr(vs, "check_raw_cache_pin", pin)
    with pytest.raises(ValueError, match="stop after the marker"):
        _rehearse(inputs, out)


def test_rehearsal_output_is_refused_by_the_search(done):
    with pytest.raises(ValueError, match="is a rehearsal folder"):
        vs.refuse_rehearsal_folder(done["out"])


def test_raw_cache_pin_and_resource_check_run_before_training(env, inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "RAW_CACHE_EVIDENCE", inputs["base"] / "missing.json")
    with pytest.raises(FileNotFoundError):
        _rehearse(inputs, tmp_path / "out")
    assert not any(c[0] in ("train", "resources") for c in env if isinstance(c, tuple))


def test_configs_unchanged_and_epoch_limits_in_code(done):
    from tests._v2_fakes import V2_CONFIG_PATH
    assert "max_epochs: 30" in V2_CONFIG_PATH.read_text(encoding="utf-8")
    assert (vr.TIMING_EPOCHS, vr.DETERMINISM_EPOCHS, vr.CRASH_EPOCH) == (1, 2, 2)


def test_module_has_no_val_b_test_or_external_path_and_no_wandb():
    text = Path(vr.__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for forbidden in ("val-B", "val_b", "test", "external"):
        assert forbidden not in strings
    assert "V2RunLogger" not in text and "import wandb" not in text
