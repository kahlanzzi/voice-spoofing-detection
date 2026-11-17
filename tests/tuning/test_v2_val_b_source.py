"""F3 waveform val-B source for an aasist risk-v2 fit (PR V4c).

Synthetic WAVs and an F2-like manifest with all four splits (tests/_v2_fakes.py), read through the
real F3 path on CPU. A read spy on stage11_v2_run (whose input pass the source uses) records every
clip read. No real data. Convention: label 1 = spoof (FAKE), 0 = bonafide (REAL).
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests._torch import require_torch

require_torch()

import src.evaluation.stage11_v2_run as m  # noqa: E402
import src.tuning.v2_val_b_source as vbs  # noqa: E402
from src.preprocessing.validation import read_verified_audio  # noqa: E402
from src.preprocessing.waveform import preprocess_waveform  # noqa: E402
from src.tuning.v2_risk_run import WAVEFORM_SAMPLES, _sha256_arrays  # noqa: E402
from tests._v2_fakes import PROFILE_PATH, profile as load_test_profile, write_dataset  # noqa: E402

SIZES = {"train": 4, "val-A": 4, "val-B": 6, "test": 4}
N = SIZES["val-B"]
SOURCE_FILE = Path(vbs.__file__)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    base = tmp_path_factory.mktemp("v2_val_b_source")
    root, manifest, manifest_path = write_dataset(base, sizes=SIZES)
    # Shuffle the manifest rows so "sorted by path" is the source's doing, not the file's order.
    manifest.sample(frac=1.0, random_state=3).to_csv(manifest_path, index=False)
    return {"base": base, "root": root, "manifest": manifest, "manifest_path": manifest_path}


@pytest.fixture()
def reads(data, monkeypatch):
    """The synthetic manifest as F2; every read through stage11_v2_run's pass recorded."""
    monkeypatch.setattr(vbs, "load_f2_manifest", lambda p: pd.read_csv(p, dtype=str, keep_default_na=False))
    seen = []

    def spy(root, row):
        seen.append(row["path"])
        return read_verified_audio(root, row)

    monkeypatch.setattr(m, "read_verified_audio", spy)
    return seen


def _source(data):
    return vbs.val_b_waveform_source(data["root"], data["manifest_path"], PROFILE_PATH)


def _files(folder: Path) -> list:
    return sorted((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in folder.rglob("*"))


def test_source_returns_exactly_the_val_b_rows_sorted_unique(data, reads):
    before = _files(data["base"])
    waves, labels, paths = _source(data)()
    manifest = data["manifest"]
    expected = sorted(manifest.loc[manifest.split == "val-B", "path"])
    assert paths.dtype.kind == "U" and paths.shape == (N,) and paths.tolist() == expected
    assert len(set(paths.tolist())) == N
    assert labels.dtype == np.int8 and labels.shape == (N,)
    by_path = manifest.set_index("path")["label"]
    assert labels.tolist() == [1 if by_path[p] == "FAKE" else 0 for p in expected]
    assert waves.dtype == np.float32 and waves.shape == (N, WAVEFORM_SAMPLES) and np.isfinite(waves).all()
    # Independently: the F3 waveform of each clip, as v2_risk_run's own aasist test builds it.
    prof = load_test_profile()
    rows = manifest.set_index("path").loc[expected].reset_index()
    by_hand = np.stack([preprocess_waveform(read_verified_audio(data["root"], row)[0], 16000, prof).waveform
                        for _, row in rows.iterrows()]).astype(np.float32)
    assert waves.tobytes() == by_hand.tobytes()
    # Byte-identical to stage11_v2_run's in-memory aasist val-B input (rehearse part (c)).
    stage11_rows = m.split_rows(pd.read_csv(data["manifest_path"], dtype=str, keep_default_na=False),
                                m.FIT_SPLIT)
    x, _ = m.compute_inputs(data["root"], stage11_rows, prof, m.WAVEFORM)
    assert _sha256_arrays(waves, labels, paths) == _sha256_arrays(x, *m._rows_arrays(stage11_rows))
    assert _files(data["base"]) == before  # nothing written next to the data (in memory only)


def test_each_val_b_clip_is_read_twice_and_nothing_else(data, reads):
    source = _source(data)
    assert reads == []  # building the source reads nothing
    source()
    manifest = data["manifest"]
    expected = sorted(manifest.loc[manifest.split == "val-B", "path"])
    assert reads == expected * 2  # two passes, val-B only: no train, val-A or test path is read
    assert all(p.startswith("val-B/") for p in reads)


def test_differing_second_pass_is_refused(data, reads, monkeypatch):
    calls = []

    def drifting(root, row):
        calls.append(row["path"])
        waveform, rate = read_verified_audio(root, row)
        if len(calls) > N:  # the second pass sees slightly different audio
            waveform = waveform + np.float32(1e-3)
        return waveform, rate

    monkeypatch.setattr(m, "read_verified_audio", drifting)
    with pytest.raises(ValueError, match="input passes differ"):
        _source(data)()
    assert len(calls) == 2 * N


def test_module_has_no_split_argument_other_split_or_disk_write():
    assert list(inspect.signature(vbs.val_b_waveform_source).parameters) == ["root", "manifest_path",
                                                                             "profile_path"]
    tree = ast.parse(SOURCE_FILE.read_text(encoding="utf-8"))
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for forbidden in ("train", "test", "external", "val-A"):
        assert forbidden not in strings
    names = {n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(tree)
             if isinstance(n, (ast.Attribute, ast.Name))}
    assert not names & {"SPLIT", "GATE_SPLIT", "open", "write_bytes", "write_text", "savez", "mkdir",
                        "_write_npz", "to_csv"}
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "src.evaluation.stage11_run" not in imported and not any("wandb" in i for i in imported)
    assert m.FIT_SPLIT == "val-B" and m.WAVEFORM == "waveform"
