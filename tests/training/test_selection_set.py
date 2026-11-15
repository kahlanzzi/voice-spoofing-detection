"""Selection condition set S (D49, D50 (4)): writer/loader round trip and every refusal. Synthetic only."""
from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

import src.training.selection_set as s_module
from src.features.logmel_cache import load_logmel_cache
from src.training.selection_set import (
    CONDITIONS,
    MANIFEST_NAME,
    condition_file,
    load_selection_set,
)
from tests._v2_fakes import (
    PROFILE_PATH,
    S_COUNTS,
    build_caches,
    manifest_fields,
    profile as load_test_profile,
    selection_arrays,
    sha,
    write_dataset,
    write_s,
)

AUG_SHA = "b" * 64


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    base = tmp_path_factory.mktemp("s")
    prof = load_test_profile()
    root, manifest, manifest_path = write_dataset(base)
    _, lm = build_caches(base, root, manifest, manifest_path, prof)
    arrays = selection_arrays(root, manifest, prof)
    f2 = sha(manifest_path)
    folder = base / "S"
    s_sha = write_s(folder, arrays, prof, f2, AUG_SHA)
    ref = load_logmel_cache(lm / "val-A", "val-A", prof, expected_f2_manifest_sha256=f2)
    return {"profile": prof, "manifest": manifest, "arrays": arrays, "f2": f2, "folder": folder,
            "sha": s_sha, "ref": ref, "base": base}


def _load(setup, folder=None, **kw):
    args = {"form": "logmel", "expected_n_rows": 10, "expected_counts": S_COUNTS,
            "recompute_rows": 2}
    args.update(kw)
    return load_selection_set(folder or setup["folder"], setup["profile"], **args)


def test_round_trip_both_forms(setup):
    for form, key in (("logmel", "log_mel"), ("waveform", "waveforms")):
        s = _load(setup, form=form, f2_manifest=setup["manifest"], clean_reference_logmel=setup["ref"],
                  recompute_rows=8)
        assert s.form == form and list(s.conditions) == list(CONDITIONS)
        assert s.sha256 == setup["sha"] == sha(setup["folder"] / MANIFEST_NAME)
        for name in CONDITIONS:
            features, labels = s.conditions[name]
            assert np.array_equal(features, setup["arrays"][name][key])
            assert np.array_equal(labels, setup["arrays"][name]["labels"])
        assert s.paths.tolist() == setup["arrays"]["clean"]["paths"].tolist()
    m = s.manifest
    assert set(m) == s_module.MANIFEST_KEYS
    assert (m["source_split"], m["seed"], m["n_rows"], m["schema_version"]) == ("val-A", 2026, 10, 1)
    assert [c["name"] for c in m["conditions"]] == list(CONDITIONS)
    assert sorted(m["files"]) == sorted(condition_file(c) for c in CONDITIONS)


def test_clean_condition_equals_stage8_val_a_cache(setup):
    features, labels, paths, _ = setup["ref"]
    clean = setup["arrays"]["clean"]
    assert features.tobytes() == clean["log_mel"].tobytes()
    assert labels.tobytes() == clean["labels"].tobytes() and paths.tolist() == clean["paths"].tolist()


def test_rewrite_is_byte_identical(setup, tmp_path):
    again = write_s(tmp_path / "S2", setup["arrays"], setup["profile"], setup["f2"], AUG_SHA)
    assert again == setup["sha"]


def test_sidecar_listed_and_verified(setup, tmp_path):
    folder = tmp_path / "S3"
    write_s(folder, setup["arrays"], setup["profile"], setup["f2"], AUG_SHA,
            sidecars={"noise_index.csv": b"path,file\n"})
    s = _load(setup, folder)
    assert "noise_index.csv" in s.manifest["files"]
    with pytest.raises(ValueError, match="plain file name"):
        write_s(tmp_path / "S4", setup["arrays"], setup["profile"], setup["f2"], AUG_SHA,
                sidecars={MANIFEST_NAME: b"x"})


def _copy(setup, tmp_path) -> Path:
    folder = tmp_path / "copy"
    shutil.copytree(setup["folder"], folder)
    return folder


def _edit_manifest(folder, change):
    m = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))
    change(m)
    (folder / MANIFEST_NAME).write_text(json.dumps(m), encoding="utf-8")


def test_refuses_missing_unlisted_and_tampered_files(setup, tmp_path):
    folder = _copy(setup, tmp_path)
    (folder / "extra.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="differ from files present"):
        _load(setup, folder)
    (folder / "extra.bin").unlink()
    (folder / condition_file("opus_16")).unlink()
    with pytest.raises(ValueError, match="differ from files present"):
        _load(setup, folder)
    shutil.copy(setup["folder"] / condition_file("opus_16"), folder / condition_file("opus_16"))
    blob = bytearray((folder / condition_file("narrowband")).read_bytes())
    blob[-50] ^= 0x01
    (folder / condition_file("narrowband")).write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="narrowband.npz: SHA-256"):
        _load(setup, folder)


@pytest.mark.parametrize("change, match", [
    (lambda m: m.update(source_split="val-B"), "source_split must be 'val-A'"),
    (lambda m: m.update(seed=2027), "seed must be 2026"),
    (lambda m: m["conditions"].reverse(), "condition names and order"),
    (lambda m: m["conditions"][1].update(name="noise_5db"), "condition names and order"),
    (lambda m: m["conditions"].pop(), "conditions must be a list of 5"),
    (lambda m: m.update(profile_fingerprint="0" * 64), "profile_fingerprint"),
    (lambda m: m.update(n_rows=11), "n_rows must be 10"),
    (lambda m: m.update(extra=1), "unknown"),
    (lambda m: m.update(runtime={"python": 3}), "runtime"),
    (lambda m: m["files"].update({"clean.npz": "0" * 64}), "clean.npz: SHA-256"),
])
def test_refuses_bad_manifest(setup, tmp_path, change, match):
    folder = _copy(setup, tmp_path)
    _edit_manifest(folder, change)
    with pytest.raises(ValueError, match=match):
        _load(setup, folder)


def _write_variant(setup, tmp_path, name, mutate, **kw):
    arrays = copy.deepcopy(setup["arrays"])
    mutate(arrays)
    folder = tmp_path / name
    # Writer refuses first; then write the bytes the writer would refuse, to test the loader too.
    with pytest.raises(ValueError) as writer_error:
        write_s(folder, arrays, setup["profile"], setup["f2"], AUG_SHA, recompute_rows=2)
    assert not folder.exists()
    shutil.copytree(setup["folder"], folder)
    files = {}
    for c in CONDITIONS:
        s_module._write_npz(folder / condition_file(c), arrays[c])
        files[condition_file(c)] = sha(folder / condition_file(c))
    _edit_manifest(folder, lambda m: m.update(files=files))
    with pytest.raises(ValueError) as loader_error:
        _load(setup, folder, **kw)
    return str(writer_error.value), str(loader_error.value)


@pytest.mark.parametrize("mutate, match", [
    (lambda a: a["noise_10db"].update(paths=a["noise_10db"]["paths"][[1, 0, *range(2, 10)]]),
     "sorted|differ"),
    (lambda a: a["opus_16"].update(labels=(1 - a["opus_16"]["labels"]).astype(np.int8)), "labels differ"),
    (lambda a: a["reverberation"].update(groups=np.array([f"x{i}" for i in range(10)])), "groups differ"),
    (lambda a: a["clean"].update(log_mel=a["clean"]["log_mel"][:, :, :32]), "log_mel must be float32"),
    (lambda a: a["clean"].update(waveforms=a["clean"]["waveforms"].astype(np.float64)), "waveforms must be float32"),
    (lambda a: a["narrowband"]["log_mel"].__setitem__((3, 0, 0, 0), np.nan), "NaN"),
    (lambda a: a["opus_16"]["log_mel"].__setitem__((0, 0, 5, 5), 1.0), "recomputed"),
])
def test_writer_and_loader_refuse_bad_arrays(setup, tmp_path, mutate, match):
    writer, loader = _write_variant(setup, tmp_path, "bad", mutate)
    assert re.search(match, writer), writer
    assert re.search(match, loader), loader


def test_refuses_path_not_val_a_and_label_or_group_mismatch_in_f2(setup):
    manifest = setup["manifest"].copy()
    first = setup["arrays"]["clean"]["paths"][0]
    for column, value, match in (("split", "val-B", "must all be val-A"),
                                 ("label", None, "labels differ from the F2"),
                                 ("group", "other", "groups differ from the F2")):
        changed = manifest.copy()
        if value is None:
            value = "REAL" if changed.loc[changed.path == first, "label"].item() == "FAKE" else "FAKE"
        changed.loc[changed.path == first, column] = value
        with pytest.raises(ValueError, match=match):
            _load(setup, f2_manifest=changed)
    with pytest.raises(ValueError, match="not in the F2 manifest"):
        _load(setup, f2_manifest=manifest[manifest.path != first])


def test_refuses_clean_reference_mismatch(setup):
    features, labels, paths, meta = setup["ref"]
    shifted = features.copy()
    shifted[0, 0, 0, 0] = np.nextafter(shifted[0, 0, 0, 0], np.float32(np.inf))
    with pytest.raises(ValueError, match="clean log-Mel differs"):
        _load(setup, clean_reference_logmel=(shifted, labels, paths, meta))
    with pytest.raises(ValueError, match="clean paths or labels differ"):
        _load(setup, clean_reference_logmel=(features, labels[::-1].copy(), paths, meta))


def test_refuses_counts_and_form(setup):
    with pytest.raises(ValueError, match="label counts"):
        _load(setup, expected_counts={"REAL": 4, "FAKE": 6})
    with pytest.raises(ValueError, match="n_rows must be 1330"):
        load_selection_set(setup["folder"], setup["profile"], form="logmel")
    with pytest.raises(ValueError, match="form must be"):
        _load(setup, form="both")


def test_writer_refusals(setup, tmp_path):
    arrays, prof = setup["arrays"], setup["profile"]
    fields = manifest_fields(prof, setup["f2"], AUG_SHA)
    with pytest.raises(ValueError, match="already exists"):
        s_module.write_selection_set(setup["folder"], arrays, fields, profile=prof)
    with pytest.raises(ValueError, match="inside the repository"):
        s_module.write_selection_set(PROFILE_PATH.parent / "s_should_not_exist", arrays, fields,
                                     profile=prof)
    reordered = {c: arrays[c] for c in reversed(CONDITIONS)}
    with pytest.raises(ValueError, match="in order"):
        s_module.write_selection_set(tmp_path / "a", reordered, fields, profile=prof)
    with pytest.raises(ValueError, match="manifest_fields"):
        s_module.write_selection_set(tmp_path / "b", arrays, {**fields, "seed": 2026}, profile=prof)
    with pytest.raises(ValueError, match="profile_fingerprint"):
        s_module.write_selection_set(tmp_path / "c", arrays, {**fields, "profile_fingerprint": "0" * 64},
                                     profile=prof)
    assert not any(tmp_path.iterdir())
