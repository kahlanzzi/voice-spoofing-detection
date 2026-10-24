"""D30 CNN search orchestrator (Stages 8-9): selection rules, train / val-A only, outputs, all or nothing.

Synthetic WAV files, manifest and log-Mel caches (seed 2026); CPU; W&B disabled.
Convention: label 1 = spoof (FAKE), 0 = bonafide (REAL).
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from tests._torch import require_torch

torch = require_torch()

import src.tuning.cnn_search as search_module  # noqa: E402
from src.evaluation.seeds import select_bundle_seed  # noqa: E402
from src.features.logmel_cache import build_logmel_cache  # noqa: E402
from src.models.cnn_checkpoint import load_checkpoint  # noqa: E402
from src.preprocessing import PreprocessingError, load_profile  # noqa: E402
from src.training.cnn_config import load_cnn_config  # noqa: E402
from src.training.train_cnn import TrainResult  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO / "configs" / "preprocessing.yaml"
CONFIG_PATH = REPO / "configs" / "cnn.yaml"
SEARCH_FILE = REPO / "src" / "tuning" / "cnn_search.py"
COMMIT = "c" * 40
SIZES = {"train": 24, "val-A": 12, "val-B": 6}  # val-B is cached but must never be read
OUTPUTS = sorted(["checkpoints", "run_report.json", "search_report.json", "selection_report.json"])
GRID = [(3, 0.001), (3, 0.0003), (4, 0.001), (4, 0.0003)]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- selection with a scripted train_one -------------------------------------------------------

def _result(eer: float, seed: int = 0) -> TrainResult:
    return TrainResult(state_dict={}, mean=np.zeros(64, np.float32), std=np.ones(64, np.float32),
                       best_epoch=1, best_val_a_eer=eer, epochs_run=1, history=(),
                       determinism={"seed": seed}, peak_gpu_memory_bytes=None)


class _NullLogger:
    finished = 0

    def finish(self):
        _NullLogger.finished += 1


def _scripted_search(monkeypatch, eers: dict):
    """Run search_and_select with train_one returning eers[(n_blocks, lr, seed)]; return calls too."""
    calls = []

    def fake_train_one(*args, n_blocks, learning_rate, seed, **kw):
        calls.append((n_blocks, learning_rate, seed))
        return _result(eers[(n_blocks, learning_rate, seed)], seed)

    monkeypatch.setattr(search_module, "train_one", fake_train_one)
    config = load_cnn_config(CONFIG_PATH)
    out = search_module.search_and_select(
        None, None, None, None, config,
        lambda n, lr, s: (_NullLogger(), search_module.run_name("v", n, lr, s)), "cpu",
    )
    return (*out, calls)


def _eers(search: dict, seeds: dict, selected) -> dict:
    out = {(n, lr, 0): v for (n, lr), v in search.items()}
    out.update({(*selected, s): v for s, v in seeds.items()})
    return out


def test_lowest_eer_wins_and_only_selected_config_gets_seeds_1_and_2(monkeypatch):
    search = {(3, 0.001): 0.2, (3, 0.0003): 0.15, (4, 0.001): 0.05, (4, 0.0003): 0.1}
    eers = _eers(search, {1: 0.07, 2: 0.03}, (4, 0.001))
    runs, selected, tied, seeds, names, calls = _scripted_search(monkeypatch, eers)
    assert selected == (4, 0.001) and tied == [(4, 0.001)]
    assert calls == [(*cfg, 0) for cfg in GRID] + [(4, 0.001, 1), (4, 0.001, 2)]
    assert seeds[0] is runs[(4, 0.001)]  # the search run is reused as seed 0
    assert {s: r.best_val_a_eer for s, r in seeds.items()} == {0: 0.05, 1: 0.07, 2: 0.03}
    assert len(names) == 6  # 4 search runs + seeds 1 and 2
    assert names[(4, 0.001, 2)] == "v-b4-lr0.001-s2"


def test_full_tie_selects_3_blocks_and_lr_0_001(monkeypatch):
    eers = _eers(dict.fromkeys(GRID, 0.0), {1: 0.0, 2: 0.0}, (3, 0.001))
    _, selected, tied, _, _, calls = _scripted_search(monkeypatch, eers)
    assert selected == (3, 0.001)
    assert tied == GRID  # tie-rule order: 3 blocks first, then the higher learning rate
    assert calls[4:] == [(3, 0.001, 1), (3, 0.001, 2)]


def test_tie_between_learning_rates_within_4_blocks_selects_0_001(monkeypatch):
    search = {(3, 0.001): 0.2, (3, 0.0003): 0.2, (4, 0.001): 0.1, (4, 0.0003): 0.1}
    eers = _eers(search, {1: 0.1, 2: 0.1}, (4, 0.001))
    _, selected, tied, _, _, _ = _scripted_search(monkeypatch, eers)
    assert selected == (4, 0.001) and tied == [(4, 0.001), (4, 0.0003)]


def test_select_configuration_tie_rule_order():
    assert search_module.select_configuration(
        {(4, 0.001): 0.1, (3, 0.0003): 0.1, (4, 0.0003): 0.1}) == ((3, 0.0003), [(3, 0.0003),
                                                                              (4, 0.001),
                                                                              (4, 0.0003)])


@pytest.mark.parametrize("seed_eers", [{0: 0.1, 1: 0.3, 2: 0.2}, {0: 0.2, 1: 0.2, 2: 0.1},
                                       {0: 0.3, 1: 0.1, 2: 0.2}])
def test_bundle_seed_follows_select_bundle_seed(monkeypatch, seed_eers):
    search = {(3, 0.001): seed_eers[0], (3, 0.0003): 0.5, (4, 0.001): 0.5, (4, 0.0003): 0.5}
    eers = _eers(search, {1: seed_eers[1], 2: seed_eers[2]}, (3, 0.001))
    _, _, _, seeds, _, _ = _scripted_search(monkeypatch, eers)
    got = select_bundle_seed({s: r.best_val_a_eer for s, r in seeds.items()})
    assert got == select_bundle_seed(seed_eers)


def test_module_has_no_val_b_test_or_external_path():
    tree = ast.parse(SEARCH_FILE.read_text(encoding="utf-8"))
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for forbidden in ("val-B", "val_b", "test", "external"):
        assert forbidden not in strings
    assert search_module.SEARCH_SPLITS == ("train", "val-A")


# --- end to end on tiny synthetic caches -------------------------------------------------------

@pytest.fixture(scope="module")
def profile():
    return load_profile(PROFILE_PATH)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory, profile):
    """Synthetic FoR-like WAVs, their manifest CSV and the train, val-A and val-B log-Mel caches."""
    base = tmp_path_factory.mktemp("cnn_search")
    rng = np.random.default_rng(2026)
    root = base / "for"
    rows = []
    specs = [(split, i % 2) for split, n in SIZES.items() for i in range(n)]
    specs.append(("test", 0))  # in the manifest only; the search never reads it
    for i, (split, spoof) in enumerate(specs):
        rate = 16000
        t = np.arange(int(rate * rng.uniform(1.6, 2.2))) / rate
        tone = np.sin(2 * np.pi * rng.uniform(150, 900) * t)
        mix = (0.7 if spoof else 0.1) + rng.uniform(-0.05, 0.05)
        data = 0.3 * ((1 - mix) * tone + mix * rng.standard_normal(len(t)))
        rel = f"{split}/{'FAKE' if spoof else 'REAL'}/{i:02d}.wav"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, data.astype(np.float32), rate, subtype="PCM_16")
        rows.append({"path": rel, "label": "FAKE" if spoof else "REAL",
                     "subgroup": "FAKE_other" if spoof else "REAL", "group": f"g{i}",
                     "split": split, "hash": _sha(path)})
    manifest = pd.DataFrame(rows, dtype=str)
    manifest_path = base / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    cache_dir = base / "cache"
    for split in SIZES:
        build_logmel_cache(root, manifest, profile, split, cache_dir / split,
                           f2_manifest_sha256=_sha(manifest_path),
                           profile_config_sha256=_sha(PROFILE_PATH), code_commit=COMMIT)
    return {"cache_dir": cache_dir, "manifest_path": manifest_path}


@pytest.fixture()
def patched(inputs, monkeypatch):
    """Clean worktree, the synthetic manifest as the approved F2 file, and tiny epochs (test only)."""
    monkeypatch.setattr(search_module, "code_version", lambda: {"git_commit": COMMIT, "git_dirty": False})
    monkeypatch.setattr(search_module, "load_f2_manifest",
                        lambda p: pd.read_csv(p, dtype=str, keep_default_na=False))
    monkeypatch.setattr(search_module, "F2_MANIFEST_SHA256", _sha(inputs["manifest_path"]))
    real = search_module.load_cnn_config
    monkeypatch.setattr(search_module, "load_cnn_config",
                        lambda p: dataclasses.replace(real(p), max_epochs=2, patience=1, batch_size=8))
    return inputs


def _run(inputs, out):
    return search_module.run_search(inputs["cache_dir"], PROFILE_PATH, CONFIG_PATH,
                                    inputs["manifest_path"], out, version="cnn-test",
                                    logger_mode="disabled", device="cpu")


def test_end_to_end_outputs_checkpoints_and_reports(patched, tmp_path, profile, monkeypatch):
    loaded = []
    real_load = search_module.load_logmel_cache

    def spy(out_dir, split, *args, **kw):
        loaded.append(split)
        return real_load(out_dir, split, *args, **kw)

    monkeypatch.setattr(search_module, "load_logmel_cache", spy)
    out = tmp_path / "run"
    report = _run(patched, out)
    assert loaded == ["train", "val-A"]  # val-B is never loaded (D42)
    assert sorted(p.name for p in out.iterdir()) == OUTPUTS
    assert sorted(p.name for p in (out / "checkpoints").iterdir()) == ["seed_0", "seed_1", "seed_2"]

    data = json.loads((out / "run_report.json").read_text(encoding="utf-8"))
    assert data == report and list(data) == sorted(data)
    assert set(data) == {
        "schema_version", "version", "code_commit", "environment", "profile_fingerprint",
        "profile_config_sha256", "f2_manifest_sha256", "cnn_config_sha256", "caches",
        "train_cache_npz_sha256", "val_a_cache_npz_sha256", "checkpoint_sha256",
        "peak_gpu_memory_bytes", "wandb_mode", "wandb_runs", "selected", "bundle_seed",
    }
    assert (data["schema_version"], data["version"], data["code_commit"]) == (1, "cnn-test", COMMIT)
    assert set(data["environment"]) == {"python", "numpy", "torch", "safetensors", "wandb",
                                        "cuda_available", "cuda_version", "cudnn_version",
                                        "gpu_name", "device"}
    assert data["environment"]["torch"] == torch.__version__ and data["environment"]["device"] == "cpu"
    assert data["profile_fingerprint"] == profile.fingerprint()
    assert data["profile_config_sha256"] == _sha(PROFILE_PATH)
    assert data["cnn_config_sha256"] == _sha(CONFIG_PATH)
    assert data["f2_manifest_sha256"] == _sha(patched["manifest_path"])
    assert set(data["caches"]) == {"train", "val-A"}
    for split, key in (("train", "train_cache_npz_sha256"), ("val-A", "val_a_cache_npz_sha256")):
        npz = patched["cache_dir"] / split / f"{split}.npz"
        assert data["caches"][split] == {"npz_sha256": _sha(npz), "n_rows": SIZES[split]}
        assert data[key] == _sha(npz)
    assert len(data["peak_gpu_memory_bytes"]) == 6
    assert all(v is None for v in data["peak_gpu_memory_bytes"].values())
    assert sorted(data["peak_gpu_memory_bytes"]) == data["wandb_runs"]
    assert data["wandb_mode"] == "disabled"

    search = json.loads((out / "search_report.json").read_text(encoding="utf-8"))
    selection = json.loads((out / "selection_report.json").read_text(encoding="utf-8"))
    assert set(search) == {"search_seed", "rows"} and search["search_seed"] == 0
    assert [(r["n_blocks"], r["learning_rate"]) for r in search["rows"]] == GRID
    for row in search["rows"]:
        assert set(row) == {"n_blocks", "learning_rate", "best_epoch", "best_val_a_eer", "epochs_run",
                            "history"}
        assert len(row["history"]) == row["epochs_run"] <= 2
        assert set(row["history"][0]) == {"epoch", "train_loss", "val_a_eer", "val_a_auc",
                                          "epoch_seconds"}
    assert set(selection) == {"selected", "selected_val_a_eer", "tie_rule", "tie_rule_decided",
                              "tied_rows", "seeds", "bundle_seed"}
    best = min(r["best_val_a_eer"] for r in search["rows"])
    assert selection["selected_val_a_eer"] == best
    assert selection["tie_rule_decided"] == (len(selection["tied_rows"]) > 1)
    assert selection["selected"] == data["selected"] == selection["tied_rows"][0]
    assert set(selection["seeds"]) == {"0", "1", "2"}
    assert [selection["seeds"][s]["reused_search_run"] for s in "012"] == [True, False, False]
    seed_eers = {int(s): v["best_val_a_eer"] for s, v in selection["seeds"].items()}
    assert selection["bundle_seed"] == data["bundle_seed"] == select_bundle_seed(seed_eers)

    for seed in (0, 1, 2):
        folder = out / "checkpoints" / f"seed_{seed}"
        model, mean, std, meta = load_checkpoint(folder, expected_sha256=data["checkpoint_sha256"][str(seed)])
        assert (meta["seed"], meta["version"], meta["code_commit"]) == (seed, "cnn-test", COMMIT)
        assert (meta["n_blocks"], meta["learning_rate"]) == (data["selected"]["n_blocks"],
                                                             data["selected"]["learning_rate"])
        assert meta["val_a_eer"] == seed_eers[seed]
        assert meta["best_epoch"] == selection["seeds"][str(seed)]["best_epoch"]
        assert meta["environment"] == data["environment"]
        assert meta["train_cache_npz_sha256"] == data["train_cache_npz_sha256"]
        assert meta["determinism"]["seed"] == seed
        assert mean.shape == std.shape == (64,)


def test_failure_leaves_nothing(patched, tmp_path, monkeypatch):
    real_save = search_module.save_checkpoint
    count = {"n": 0}

    def failing(*args, **kw):
        count["n"] += 1
        if count["n"] == 2:
            raise OSError("disk full")
        return real_save(*args, **kw)

    monkeypatch.setattr(search_module, "save_checkpoint", failing)
    out = tmp_path / "run"
    with pytest.raises(OSError, match="disk full"):
        _run(patched, out)
    assert not out.exists()


def test_training_failure_writes_nothing(patched, tmp_path, monkeypatch):
    def boom(*args, **kw):
        raise RuntimeError("diverged")

    monkeypatch.setattr(search_module, "train_one", boom)
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="diverged"):
        _run(patched, out)
    assert not out.exists()


def test_existing_out_dir_is_refused(patched, tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        _run(patched, out)
    assert list(out.iterdir()) == []


def test_dirty_worktree_is_refused_before_reading(inputs, tmp_path, monkeypatch):
    def dirty():
        raise PreprocessingError("Official Stage 6 validation requires a clean Git worktree")

    def must_not_run(*args, **kw):
        raise AssertionError("input read before the code version check")

    monkeypatch.setattr(search_module, "code_version", dirty)
    for name in ("load_cnn_config", "load_profile", "load_f2_manifest", "load_logmel_cache"):
        monkeypatch.setattr(search_module, name, must_not_run)
    out = tmp_path / "run"
    with pytest.raises(PreprocessingError, match="clean Git worktree"):
        _run(inputs, out)
    assert not out.exists()


@pytest.mark.parametrize("change,match", [
    ({"logger_mode": "dryrun"}, "logger_mode"),
    ({"version": " "}, "version"),
])
def test_invalid_arguments_are_refused(patched, tmp_path, change, match):
    kw = {"version": "cnn-test", "logger_mode": "disabled", "device": "cpu", **change}
    with pytest.raises(ValueError, match=match):
        search_module.run_search(patched["cache_dir"], PROFILE_PATH, CONFIG_PATH,
                                 patched["manifest_path"], tmp_path / "run", **kw)
    assert not (tmp_path / "run").exists()


def test_cli_parses_required_arguments(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_run(*args, **kw):
        seen.update(args=args, kw=kw)
        return {"ok": True}

    monkeypatch.setattr(search_module, "run_search", fake_run)
    search_module.main(["--cache-dir", "c", "--profile", "p", "--config", "f", "--manifest", "m",
                        "--out", str(tmp_path / "o"), "--version", "cnn-v1", "--wandb-mode",
                        "offline", "--device", "cuda"])
    assert seen["kw"] == {"version": "cnn-v1", "logger_mode": "offline", "device": "cuda"}
    assert json.loads(capsys.readouterr().out) == {"ok": True}
    with pytest.raises(SystemExit):
        search_module.main(["--cache-dir", "c"])
