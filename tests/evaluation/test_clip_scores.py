"""Per-clip score files (D41, D45) on synthetic rows and temporary folders only (no dataset access)."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from src.evaluation.clip_scores import (
    APPLICABLE_COLUMNS,
    COLUMNS,
    SIDECAR_KEYS,
    read_clip_scores,
    write_clip_scores,
)

RUN_ID = "0123456789abcdef0123456789abcdef"
SHA_A = "a" * 64
SHA_B = "b" * 64
CLIPS = ("synth/fake/f2.wav", "synth/real/r1.wav", "synth/fake/f1.wav", "synth/real/r 2,x.wav")
LABELS = ("FAKE", "REAL", "FAKE", "REAL")
GROUPS = ("17", "3", "17", "42")
RAW = (1.5, -2.25, 0.1, -1e-300)
P = (0.8175744761936437, 0.09534945969074979, 0.52, 0.0)
PRED = {
    "baseline": ("synthetic", "authentic", "synthetic", "authentic"),
    "cnn_bundle_seed": ("synthetic", "authentic", "uncertain", "authentic"),
}


def _args(role, **change):
    args = {
        "evaluation_purpose": "stage11_test",
        "run_id": RUN_ID,
        "model_id": f"{role}-m",
        "model_role": role,
        "model_artifact_sha256": SHA_A,
        "manifest_sha256": SHA_B,
        "clip_id": CLIPS,
        "label": LABELS,
        "group_id": GROUPS,
        "raw_score": RAW,
        "p": P if "p" in APPLICABLE_COLUMNS[role] else None,
        "prediction": PRED[role] if "prediction" in APPLICABLE_COLUMNS[role] else None,
    }
    args.update(change)
    return args


def _write(folder, role, **change):
    folder.mkdir(parents=True, exist_ok=True)
    return write_clip_scores(folder, **_args(role, **change))


ROLES = sorted(APPLICABLE_COLUMNS)


def test_applicability_table_is_d41_d45():
    # Fixed in code from D41 (baseline: raw score + D29 prediction, no p) and D45 (seeds 1-2:
    # raw score only); this test fails if the table drifts without a decision.
    assert COLUMNS == ("clip_id", "label", "group_id", "raw_score", "p", "prediction")
    assert APPLICABLE_COLUMNS == {
        "baseline": ("clip_id", "label", "group_id", "raw_score", "prediction"),
        "cnn_bundle_seed": ("clip_id", "label", "group_id", "raw_score", "p", "prediction"),
        "cnn_other_seed": ("clip_id", "label", "group_id", "raw_score"),
    }


@pytest.mark.parametrize("role", ROLES)
def test_round_trip_and_format(tmp_path, role):
    # Header, LF, rows sorted by clip_id, empty cells where not applicable; the reader returns the
    # same values (floats bit-exact).
    csv_path, json_path = _write(tmp_path, role)
    raw = csv_path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    lines = raw.decode().split("\n")[:-1]
    assert lines[0] == "clip_id,label,group_id,raw_score,p,prediction"
    assert len(lines) == 1 + len(CLIPS)
    meta = json.loads(json_path.read_bytes())
    assert set(meta) == SIDECAR_KEYS and meta["n_rows"] == len(CLIPS)
    assert meta["csv_sha256"] == hashlib.sha256(raw).hexdigest()

    got = read_clip_scores(tmp_path, f"{role}-m")
    order = np.argsort(CLIPS)
    assert got.clip_id == tuple(sorted(CLIPS))
    assert got.label == tuple(LABELS[i] for i in order)
    assert got.group_id == tuple(GROUPS[i] for i in order)
    assert got.raw_score.dtype == np.float64
    assert got.raw_score.tobytes() == np.array(RAW, dtype=np.float64)[order].tobytes()
    if "p" in APPLICABLE_COLUMNS[role]:
        assert got.p.tobytes() == np.array(P, dtype=np.float64)[order].tobytes()
    else:
        assert got.p is None
        assert all(line.split(",")[4] == "" for line in lines[1:] if '"' not in line)
    if "prediction" in APPLICABLE_COLUMNS[role]:
        assert got.prediction == tuple(PRED[role][i] for i in order)
    else:
        assert got.prediction is None


@pytest.mark.parametrize("role", ROLES)
def test_byte_identical_and_order_independent(tmp_path, role):
    # Two writes of the same rows, one in reversed input order, give the same bytes (so the same
    # SHA-256 can be committed and re-checked).
    a = _write(tmp_path / "a", role)
    b = _write(tmp_path / "b", role)
    rev = {k: tuple(reversed(v)) for k, v in _args(role).items()
           if k in ("clip_id", "label", "group_id", "raw_score", "p", "prediction") and v is not None}
    c = _write(tmp_path / "c", role, **rev)
    for x, y, z in zip(a, b, c):
        assert x.read_bytes() == y.read_bytes() == z.read_bytes()


def test_float_round_trip_exact(tmp_path):
    # repr(float64) round-trips exactly, including float32-derived values (float(np.float32(x))
    # is exact) and extreme magnitudes.
    rng = np.random.default_rng(0)
    raw32 = rng.normal(size=200).astype(np.float32)
    raw = np.concatenate([raw32.astype(np.float64), [5e-324, -1.7976931348623157e308, -0.0, 1e16 + 2]])
    n = len(raw)
    p = np.concatenate([rng.random(n - 2), [0.0, 1.0]])
    clips = [f"c/{i:04d}.wav" for i in range(n)]
    write_clip_scores(
        tmp_path, **_args("cnn_bundle_seed", clip_id=clips, label=["FAKE"] * n, group_id=["g"] * n,
                          raw_score=list(raw32) + list(raw[200:]), p=p, prediction=["uncertain"] * n)
    )
    got = read_clip_scores(tmp_path, "cnn_bundle_seed-m")
    assert got.raw_score.tobytes() == raw.tobytes()
    assert got.p.tobytes() == p.tobytes()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), True, "1.0", None])
def test_bad_raw_score_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="raw_score"):
        _write(tmp_path, "cnn_other_seed", raw_score=(bad,) + RAW[1:])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad", [-0.01, 1.0000001, float("nan")])
def test_p_outside_unit_interval_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="p"):
        _write(tmp_path, "cnn_bundle_seed", p=(bad,) + P[1:])


@pytest.mark.parametrize(
    "role, change, match",
    [
        ("baseline", {"p": P}, "p is not applicable"),
        ("baseline", {"p": ("",) * 4}, "p is not applicable"),
        ("baseline", {"prediction": None}, "prediction is applicable"),
        ("cnn_other_seed", {"p": P}, "p is not applicable"),
        ("cnn_other_seed", {"prediction": PRED["baseline"]}, "prediction is not applicable"),
        ("cnn_bundle_seed", {"p": None}, "p is applicable"),
        ("cnn_bundle_seed", {"prediction": None}, "prediction is applicable"),
        ("cnn_bundle_seed", {"prediction": ("synthetic", "", "authentic", "authentic")}, "prediction is applicable"),
    ],
)
def test_writer_applicability(tmp_path, role, change, match):
    # A value in a not-applicable column, or a missing applicable column, is refused (D41, D45).
    with pytest.raises(ValueError, match=match):
        _write(tmp_path, role, **change)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "role, change, match",
    [
        ("baseline", {"prediction": ("uncertain",) + PRED["baseline"][1:]}, "prediction"),  # D29: never uncertain
        ("baseline", {"prediction": ("FAKE",) + PRED["baseline"][1:]}, "prediction"),
        ("cnn_bundle_seed", {"prediction": ("Synthetic",) + PRED["cnn_bundle_seed"][1:]}, "prediction"),
        ("cnn_other_seed", {"label": ("fake",) + LABELS[1:]}, "label"),
        ("cnn_other_seed", {"label": (1,) + LABELS[1:]}, "label"),
        ("cnn_other_seed", {"label": ("bonafide",) + LABELS[1:]}, "label"),
        ("cnn_other_seed", {"group_id": ("",) + GROUPS[1:]}, "group_id"),
        ("cnn_other_seed", {"group_id": (17,) + GROUPS[1:]}, "group_id"),
        ("cnn_other_seed", {"clip_id": (CLIPS[1],) + CLIPS[1:]}, "duplicate clip_id"),
        ("cnn_other_seed", {"clip_id": ("",) + CLIPS[1:]}, "clip_id"),
        ("cnn_other_seed", {"clip_id": ("/abs/x.wav",) + CLIPS[1:]}, "relative"),
        ("cnn_other_seed", {"clip_id": ("C:/x.wav",) + CLIPS[1:]}, "relative"),
        ("cnn_other_seed", {"clip_id": ("a/../x.wav",) + CLIPS[1:]}, "unsafe"),
        ("cnn_other_seed", {"clip_id": ("a//x.wav",) + CLIPS[1:]}, "unsafe|canonical"),
        ("cnn_other_seed", {"clip_id": ("a\\x.wav",) + CLIPS[1:]}, "canonical"),
        ("cnn_other_seed", {"clip_id": ("a/x\n.wav",) + CLIPS[1:]}, "line break"),
        ("cnn_other_seed", {"clip_id": CLIPS[:3]}, "one per clip"),
        ("cnn_other_seed", {"clip_id": (), "label": (), "group_id": (), "raw_score": ()}, "at least one"),
        ("cnn_other_seed", {"evaluation_purpose": "stage13_external"}, "new decision"),
        ("cnn_other_seed", {"evaluation_purpose": "Stage11_test"}, "new decision"),
        ("cnn_other_seed", {"run_id": "x" * 32}, "run_id"),
        ("cnn_other_seed", {"model_artifact_sha256": "A" * 64}, "model_artifact_sha256"),
        ("cnn_other_seed", {"manifest_sha256": "a" * 63}, "manifest_sha256"),
        ("cnn_other_seed", {"model_id": "../x"}, "model_id"),
        ("cnn_other_seed", {"model_role": "cnn"}, "model_role"),
    ],
)
def test_writer_validation(tmp_path, role, change, match):
    with pytest.raises(ValueError, match=match):
        _write(tmp_path, role, **change)
    assert list(tmp_path.iterdir()) == []


def test_never_overwrites(tmp_path):
    _write(tmp_path, "baseline")
    with pytest.raises(ValueError, match="never overwritten"):
        _write(tmp_path, "baseline")
    with pytest.raises(ValueError, match="does not exist"):
        write_clip_scores(tmp_path / "missing", **_args("baseline"))


def test_sidecar_created_after_validation_is_refused(tmp_path):
    # Exclusive creation ("xb"): a sidecar that appears before the writer creates it (here
    # pre-created, as if by a concurrent writer) is refused with the same error, its content is
    # untouched, and the CSV this call created is removed.
    sidecar = tmp_path / "baseline-m.json"
    sidecar.write_bytes(b"other writer")
    with pytest.raises(ValueError, match="never overwritten"):
        _write(tmp_path, "baseline")
    assert sidecar.read_bytes() == b"other writer"
    assert not (tmp_path / "baseline-m.csv").exists()


# --- reader: tampered files ------------------------------------------------------------------------


def _resign(folder, model_id, csv_bytes, **meta_change):
    # Rewrite the CSV and give the sidecar a matching csv_sha256, so the reader's own row checks
    # (not the hash) must catch the problem.
    (folder / f"{model_id}.csv").write_bytes(csv_bytes)
    meta = json.loads((folder / f"{model_id}.json").read_bytes())
    meta["csv_sha256"] = hashlib.sha256(csv_bytes).hexdigest()
    meta.update(meta_change)
    (folder / f"{model_id}.json").write_bytes(
        (json.dumps(meta, sort_keys=True, indent=2) + "\n").encode()
    )


def _rows(folder, model_id):
    return (folder / f"{model_id}.csv").read_bytes().decode().split("\n")


def _set_cell(folder, model_id, row, column, value):
    lines = _rows(folder, model_id)
    cells = lines[row].split(",")
    cells[COLUMNS.index(column)] = value
    lines[row] = ",".join(cells)
    _resign(folder, model_id, "\n".join(lines).encode())


def test_tampered_csv_byte_detected(tmp_path):
    csv_path, _ = _write(tmp_path, "cnn_bundle_seed")
    data = bytearray(csv_path.read_bytes())
    data[-3] ^= 0x01  # one byte of the last row
    csv_path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="SHA-256"):
        read_clip_scores(tmp_path, "cnn_bundle_seed-m")


@pytest.mark.parametrize(
    "role, column, value, match",
    [
        ("baseline", "p", "0.5", "p is not applicable"),
        ("cnn_other_seed", "p", "0.5", "p is not applicable"),
        ("cnn_other_seed", "prediction", "synthetic", "prediction is not applicable"),
        ("baseline", "prediction", "", "prediction is applicable"),
        ("cnn_bundle_seed", "p", "", "p is applicable"),
        ("cnn_bundle_seed", "prediction", "", "prediction is applicable"),
        ("cnn_other_seed", "raw_score", "", "raw_score"),
        ("cnn_other_seed", "raw_score", "nan", "finite"),
        ("cnn_other_seed", "raw_score", "inf", "finite"),
        ("cnn_other_seed", "raw_score", "1.50", "exact spelling"),
        ("cnn_other_seed", "raw_score", "abc", "not a number"),
        ("cnn_bundle_seed", "p", "1.5", r"\[0, 1\]"),
        ("baseline", "prediction", "uncertain", "prediction"),
        ("cnn_other_seed", "label", "REAL ", "label"),
        ("cnn_other_seed", "group_id", "", "group_id"),
        ("cnn_other_seed", "clip_id", "../x.wav", "unsafe"),
    ],
)
def test_reader_refuses_bad_cell(tmp_path, role, column, value, match):
    # Line 1 is the first data row, a clip without quoting (sorted first: synth/fake/f1.wav).
    _write(tmp_path, role)
    _set_cell(tmp_path, f"{role}-m", 1, column, value)
    with pytest.raises(ValueError, match=match):
        read_clip_scores(tmp_path, f"{role}-m")


def test_reader_refuses_crlf(tmp_path):
    csv_path, _ = _write(tmp_path, "baseline")
    _resign(tmp_path, "baseline-m", csv_path.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="CR found"):
        read_clip_scores(tmp_path, "baseline-m")


def test_reader_refuses_unsorted_or_duplicate_rows(tmp_path):
    _write(tmp_path, "cnn_other_seed")
    lines = _rows(tmp_path, "cnn_other_seed-m")
    swapped = [lines[0], lines[2], lines[1], *lines[3:]]
    _resign(tmp_path, "cnn_other_seed-m", "\n".join(swapped).encode())
    with pytest.raises(ValueError, match="exact format"):
        read_clip_scores(tmp_path, "cnn_other_seed-m")
    dup = [lines[0], lines[1], lines[1], *lines[3:]]
    _resign(tmp_path, "cnn_other_seed-m", "\n".join(dup).encode())
    with pytest.raises(ValueError, match="duplicate clip_id"):
        read_clip_scores(tmp_path, "cnn_other_seed-m")


def test_reader_refuses_header_count_and_truncation(tmp_path):
    csv_path, _ = _write(tmp_path, "cnn_other_seed")
    good = csv_path.read_bytes()
    _resign(tmp_path, "cnn_other_seed-m", good.replace(b"group_id", b"group", 1))
    with pytest.raises(ValueError, match="header"):
        read_clip_scores(tmp_path, "cnn_other_seed-m")
    _resign(tmp_path, "cnn_other_seed-m", good[:-1])
    with pytest.raises(ValueError, match="newline-terminated"):
        read_clip_scores(tmp_path, "cnn_other_seed-m")
    _resign(tmp_path, "cnn_other_seed-m", good, n_rows=3)
    with pytest.raises(ValueError, match="n_rows"):
        read_clip_scores(tmp_path, "cnn_other_seed-m")
    lines = good.decode().split("\n")
    _resign(tmp_path, "cnn_other_seed-m", "\n".join(lines[:-2] + [""]).encode(), n_rows=4)
    with pytest.raises(ValueError, match="n_rows"):  # last row dropped, sidecar count unchanged
        read_clip_scores(tmp_path, "cnn_other_seed-m")


@pytest.mark.parametrize(
    "change, match",
    [
        ({"extra": 1}, "unknown keys"),
        ({"evaluation_purpose": "stage14"}, "new decision"),
        ({"model_role": "cnn_other_seed"}, "applicable_columns"),  # role swapped, columns kept
        ({"applicable_columns": ["clip_id", "label", "group_id", "raw_score"]}, "applicable_columns"),
        ({"run_id": "A" * 32}, "run_id"),
        ({"n_rows": True}, "n_rows"),
        ({"n_rows": 4.0}, "n_rows"),
        ({"model_id": "other"}, "file name"),
        ({"manifest_sha256": "x"}, "manifest_sha256"),
    ],
)
def test_reader_refuses_bad_sidecar(tmp_path, change, match):
    _, json_path = _write(tmp_path, "baseline")
    meta = json.loads(json_path.read_bytes())
    meta.update(change)
    json_path.write_bytes((json.dumps(meta, sort_keys=True, indent=2) + "\n").encode())
    with pytest.raises(ValueError, match=match):
        read_clip_scores(tmp_path, "baseline-m")


@pytest.mark.parametrize("key", sorted(SIDECAR_KEYS))
def test_reader_refuses_missing_sidecar_key(tmp_path, key):
    _, json_path = _write(tmp_path, "cnn_bundle_seed")
    meta = json.loads(json_path.read_bytes())
    del meta[key]
    json_path.write_bytes((json.dumps(meta, sort_keys=True, indent=2) + "\n").encode())
    with pytest.raises(ValueError, match="missing keys"):
        read_clip_scores(tmp_path, "cnn_bundle_seed-m")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.replace(b"\n", b"\r\n"),  # CRLF sidecar
        lambda b: b.replace(b'"n_rows": 4', b'"n_rows": NaN'),  # non-strict JSON
        lambda b: json.dumps(json.loads(b)).encode() + b"\n",  # not indented / sorted format
    ],
)
def test_reader_refuses_non_canonical_sidecar(tmp_path, mutate):
    _, json_path = _write(tmp_path, "cnn_bundle_seed")
    json_path.write_bytes(mutate(json_path.read_bytes()))
    with pytest.raises(ValueError):
        read_clip_scores(tmp_path, "cnn_bundle_seed-m")
