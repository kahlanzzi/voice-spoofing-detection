"""Per-clip score files for one registered test-split run (D41, D45; Protocol v1.0 §7).

Stage 11 stores clean per-clip outputs once, so Stage 14 can pair degraded scores with them by clip
ID (no second clean pass, D41) and Stages 15-16 (H7, D43) can read them without a new test run (D31).
Each model of a run gets two files in a folder passed by the caller:

- <model_id>.csv: columns clip_id, label, group_id, raw_score, p, prediction, in this order. Header
  line, one row per clip sorted by clip_id, UTF-8, LF line endings ("\\n", also after the last row).
  clip_id is the F2 manifest path (a canonical relative POSIX path, the rule of
  src.freezing.manifests.safe_relative_path); label is the manifest label (FAKE / REAL); group_id is
  the D14 near-duplicate group (F2 manifest column "group"). Floats are repr(float(x)) of the
  float64 value, which round-trips exactly; NaN and inf are refused.
- <model_id>.json: the sidecar (strict JSON, sorted keys): evaluation_purpose, run_id,
  model_id, model_role, model_artifact_sha256, manifest_sha256, n_rows, csv_sha256 and
  applicable_columns.

Applicability is fixed here by model role (D41, D45), never read from the file:
- "baseline": raw_score + prediction (D29 forced binary, authentic / synthetic); p not applicable.
- "cnn_bundle_seed": raw_score + p (calibrated, F4 policy) + prediction (authentic / synthetic /
  uncertain).
- "cnn_other_seed": raw_score only; p and prediction not applicable (D45: no risk policy).
A not-applicable column is the empty string in every row. The writer refuses a value there; the
reader refuses a non-empty cell there and an empty cell in an applicable column.

Both files are written as bytes (no newline translation, same reason as src.tuning.risk_run), so
their bytes and SHA-256 are the same on every OS. The writer never overwrites a file: each is
created with open(path, "xb"), so the OS refuses an existing file atomically. The reader
re-checks every rule, recomputes csv_sha256 and n_rows, and finally requires the bytes to be
exactly what the writer would produce, so a hand-edited file (CRLF, other quoting, another float
spelling, unsorted rows) is refused.

The files do not prove that run_id is a registered run; the caller (the Stage 11 runner) takes it
from RunRegistry.claim. All functions raise ValueError on invalid input (AGENTS.md: fail loudly).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from numbers import Real
from os import PathLike
from pathlib import Path

import numpy as np

from src.evaluation.run_registry import EVALUATION_PURPOSES
from src.freezing.manifests import safe_relative_path
from src.risk.decisions import AUTHENTIC, DECISIONS, SYNTHETIC

COLUMNS = ("clip_id", "label", "group_id", "raw_score", "p", "prediction")
LABELS = frozenset({"FAKE", "REAL"})  # F2 manifest labels; positive class = spoof = FAKE
# Optional columns that apply per role (D41, D45); clip_id, label, group_id, raw_score always apply.
_OPTIONAL = {
    "baseline": ("prediction",),
    "cnn_bundle_seed": ("p", "prediction"),
    "cnn_other_seed": (),
}
APPLICABLE_COLUMNS = {
    role: tuple(c for c in COLUMNS if c not in ("p", "prediction") or c in optional)
    for role, optional in _OPTIONAL.items()
}
MODEL_ROLES = frozenset(APPLICABLE_COLUMNS)
# Prediction values per role: D29 forced binary for the baseline; three-way for the bundle seed.
PREDICTIONS = {
    "baseline": frozenset({AUTHENTIC, SYNTHETIC}),
    "cnn_bundle_seed": frozenset(DECISIONS),
}
SIDECAR_KEYS = frozenset({
    "evaluation_purpose", "run_id", "model_id", "model_role", "model_artifact_sha256",
    "manifest_sha256", "n_rows", "csv_sha256", "applicable_columns",
})

_HEX32 = re.compile(r"[0-9a-f]{32}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # also a safe file name


@dataclass(frozen=True)
class ClipScores:
    """One model's verified per-clip outputs; p / prediction are None when not applicable."""

    metadata: dict
    clip_id: tuple[str, ...]
    label: tuple[str, ...]
    group_id: tuple[str, ...]
    raw_score: np.ndarray
    p: np.ndarray | None
    prediction: tuple[str, ...] | None


def _check_choice(name: str, value, allowed) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    return value


def _check_hex(name: str, value, pattern: re.Pattern) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase hex digest, got {value!r}")
    return value


def _check_purpose(value) -> str:
    if not isinstance(value, str) or value not in EVALUATION_PURPOSES:
        raise ValueError(
            f"evaluation_purpose must be one of {sorted(EVALUATION_PURPOSES)}, got {value!r}; "
            "a new purpose needs a new decision (D41)"
        )
    return value


def _check_model_id(value) -> str:
    if not isinstance(value, str) or not _MODEL_ID.fullmatch(value):
        raise ValueError(f"model_id must match {_MODEL_ID.pattern} (a plain file name), got {value!r}")
    return value


def _check_text(name: str, row: int, value) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"row {row}: {name} must be a non-empty trimmed string, got {value!r}")
    if "\r" in value or "\n" in value:
        raise ValueError(f"row {row}: {name} must not contain a line break, got {value!r}")
    return value


def _check_clip_id(row: int, value) -> str:
    _check_text("clip_id", row, value)
    try:
        canonical = safe_relative_path(value)
    except ValueError as error:
        raise ValueError(f"row {row}: {error}") from None
    if canonical != value:
        # Pairing is by exact string, so only the canonical spelling is allowed.
        raise ValueError(f"row {row}: clip_id must be the canonical POSIX path {canonical!r}, got {value!r}")
    return value


def _float_text(name: str, row: int, value) -> str:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, np.floating, np.integer)):
        raise ValueError(f"row {row}: {name} must be a real number, got {value!r}")
    x = float(value)  # exact for float32 / float64 / int values up to 2**53
    if not math.isfinite(x):
        raise ValueError(f"row {row}: {name} must be finite, got {value!r}")
    if name == "p" and not 0.0 <= x <= 1.0:
        raise ValueError(f"row {row}: p must be in [0, 1], got {value!r}")
    return repr(x)


def _column(name: str, values, n: int | None) -> list:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of values, not a single string")
    try:
        values = list(values)
    except TypeError:
        raise ValueError(f"{name} must be a sequence, got {values!r}") from None
    if n is not None and len(values) != n:
        raise ValueError(f"{name} has {len(values)} values, expected {n} (one per clip)")
    return values


def _csv_bytes(rows: list[tuple[str, ...]]) -> bytes:
    """Header + rows sorted by clip_id, LF endings, minimal quoting, UTF-8."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    writer.writerows(sorted(rows))  # clip_id is first and unique, so this sorts by clip_id
    return buffer.getvalue().encode("utf-8")


def _sidecar_bytes(meta: dict) -> bytes:
    return (json.dumps(meta, sort_keys=True, indent=2, allow_nan=False, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _check_rows(role: str, rows: list[tuple[str, ...]]) -> None:
    """The row rules shared by writer and reader; cells are already text."""
    if not rows:
        raise ValueError("a score file needs at least one clip")
    applicable = APPLICABLE_COLUMNS[role]
    seen: set[str] = set()
    for number, row in enumerate(rows, start=1):
        cells = dict(zip(COLUMNS, row))
        clip_id = _check_clip_id(number, cells["clip_id"])
        if clip_id in seen:
            raise ValueError(f"row {number}: duplicate clip_id {clip_id!r}")
        seen.add(clip_id)
        _check_choice(f"row {number}: label", cells["label"], LABELS)
        _check_text("group_id", number, cells["group_id"])
        for name in ("p", "prediction"):
            if name not in applicable and cells[name] != "":
                raise ValueError(
                    f"row {number}: {name} is not applicable for model_role {role!r} (D41, D45) "
                    f"and must be empty, got {cells[name]!r}"
                )
            if name in applicable and cells[name] == "":
                raise ValueError(f"row {number}: {name} is applicable for model_role {role!r} and is empty")
        if "prediction" in applicable:
            _check_choice(f"row {number}: prediction", cells["prediction"], PREDICTIONS[role])


def write_clip_scores(
    out_dir: str | PathLike[str],
    *,
    evaluation_purpose: str,
    run_id: str,
    model_id: str,
    model_role: str,
    model_artifact_sha256: str,
    manifest_sha256: str,
    clip_id,
    label,
    group_id,
    raw_score,
    p=None,
    prediction=None,
) -> tuple[Path, Path]:
    """Write <model_id>.csv and <model_id>.json into an existing folder; return both paths.

    Columns are passed as sequences of equal length. p and prediction must be None exactly when
    they are not applicable for model_role (see the module docstring). Existing files are refused.
    """
    _check_purpose(evaluation_purpose)
    _check_hex("run_id", run_id, _HEX32)
    _check_model_id(model_id)
    role = _check_choice("model_role", model_role, MODEL_ROLES)
    _check_hex("model_artifact_sha256", model_artifact_sha256, _HEX64)
    _check_hex("manifest_sha256", manifest_sha256, _HEX64)
    applicable = APPLICABLE_COLUMNS[role]
    for name, values in (("p", p), ("prediction", prediction)):
        if name not in applicable and values is not None:
            raise ValueError(f"{name} is not applicable for model_role {role!r} (D41, D45); pass None")
        if name in applicable and values is None:
            raise ValueError(f"{name} is applicable for model_role {role!r} and is required")

    clip_ids = _column("clip_id", clip_id, None)
    n = len(clip_ids)
    labels = _column("label", label, n)
    groups = _column("group_id", group_id, n)
    raw = [_float_text("raw_score", i, v) for i, v in enumerate(_column("raw_score", raw_score, n), 1)]
    p_text = [""] * n if p is None else [_float_text("p", i, v) for i, v in enumerate(_column("p", p, n), 1)]
    pred = [""] * n if prediction is None else _column("prediction", prediction, n)
    for name, values in (("clip_id", clip_ids), ("label", labels), ("group_id", groups), ("prediction", pred)):
        for number, value in enumerate(values, start=1):
            if not isinstance(value, str):
                raise ValueError(f"row {number}: {name} must be a string, got {value!r}")
    rows = list(zip(clip_ids, labels, groups, raw, p_text, pred))
    _check_rows(role, rows)

    csv_bytes = _csv_bytes(rows)
    sidecar = _sidecar_bytes({
        "evaluation_purpose": evaluation_purpose,
        "run_id": run_id,
        "model_id": model_id,
        "model_role": role,
        "model_artifact_sha256": model_artifact_sha256,
        "manifest_sha256": manifest_sha256,
        "n_rows": n,
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "applicable_columns": list(applicable),
    })

    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise ValueError(f"output folder {out_dir} does not exist")
    csv_path, json_path = out_dir / f"{model_id}.csv", out_dir / f"{model_id}.json"
    _create(csv_path, csv_bytes)
    try:
        _create(json_path, sidecar)
    except BaseException:
        csv_path.unlink(missing_ok=True)  # ours: _create just made it
        raise
    return csv_path, json_path


def _create(path: Path, data: bytes) -> None:
    # "xb": the OS creates the file only if it does not exist, so "never overwrite" holds even if
    # another writer creates it after validation. Binary mode: no newline translation, so the bytes
    # (and SHA-256) are the same on every OS (write_text writes CRLF on Windows).
    try:
        with open(path, "xb") as handle:
            handle.write(data)
    except FileExistsError:
        raise ValueError(f"{path} exists; score files are never overwritten") from None


def _reject_constant(token: str):
    raise ValueError(f"sidecar is not strict JSON: {token} is not allowed")


def _read_sidecar(json_path: Path) -> dict:
    raw = json_path.read_bytes()
    try:
        meta = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{json_path}: not valid UTF-8 JSON ({error})") from None
    if not isinstance(meta, dict):
        raise ValueError(f"{json_path}: expected a JSON object")
    keys = set(meta)
    if keys != SIDECAR_KEYS:
        raise ValueError(
            f"{json_path}: unknown keys {sorted(keys - SIDECAR_KEYS)}, "
            f"missing keys {sorted(SIDECAR_KEYS - keys)}"
        )
    _check_purpose(meta["evaluation_purpose"])
    _check_hex("run_id", meta["run_id"], _HEX32)
    _check_model_id(meta["model_id"])
    role = _check_choice("model_role", meta["model_role"], MODEL_ROLES)
    _check_hex("model_artifact_sha256", meta["model_artifact_sha256"], _HEX64)
    _check_hex("manifest_sha256", meta["manifest_sha256"], _HEX64)
    _check_hex("csv_sha256", meta["csv_sha256"], _HEX64)
    if isinstance(meta["n_rows"], bool) or not isinstance(meta["n_rows"], int) or meta["n_rows"] < 1:
        raise ValueError(f"n_rows must be an int >= 1, got {meta['n_rows']!r}")
    if meta["applicable_columns"] != list(APPLICABLE_COLUMNS[role]):
        raise ValueError(
            f"applicable_columns {meta['applicable_columns']!r} differ from the fixed columns "
            f"{list(APPLICABLE_COLUMNS[role])} of model_role {role!r} (D41, D45)"
        )
    if _sidecar_bytes(meta) != raw:
        raise ValueError(f"{json_path}: not in the writer's exact format (sorted keys, indent 2, LF)")
    return meta


def read_clip_scores(out_dir: str | PathLike[str], model_id: str) -> ClipScores:
    """Read and verify <model_id>.csv against <model_id>.json; ValueError on any mismatch."""
    _check_model_id(model_id)
    out_dir = Path(out_dir)
    csv_path, json_path = out_dir / f"{model_id}.csv", out_dir / f"{model_id}.json"
    meta = _read_sidecar(json_path)
    if meta["model_id"] != model_id:
        raise ValueError(f"sidecar model_id {meta['model_id']!r} differs from the file name {model_id!r}")
    role = meta["model_role"]

    raw = csv_path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != meta["csv_sha256"]:
        raise ValueError(f"{csv_path}: SHA-256 {actual_sha} differs from the sidecar's {meta['csv_sha256']}")
    if b"\r" in raw:
        raise ValueError(f"{csv_path}: CR found; the format uses LF line endings only")
    if not raw.endswith(b"\n"):
        raise ValueError(f"{csv_path}: last line is not newline-terminated (truncated write?)")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{csv_path}: not valid UTF-8 ({error})") from None
    try:
        table = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise ValueError(f"{csv_path}: not valid CSV ({error})") from None
    if not table or tuple(table[0]) != COLUMNS:
        raise ValueError(f"{csv_path}: header must be {','.join(COLUMNS)}")
    rows = []
    for number, row in enumerate(table[1:], start=1):
        if len(row) != len(COLUMNS):
            raise ValueError(f"row {number}: expected {len(COLUMNS)} cells, got {len(row)}")
        rows.append(tuple(row))
    if len(rows) != meta["n_rows"]:
        raise ValueError(f"{csv_path}: {len(rows)} rows, the sidecar records n_rows {meta['n_rows']}")
    _check_rows(role, rows)
    floats = {}
    for name in ("raw_score", "p"):
        if name not in APPLICABLE_COLUMNS[role]:
            continue
        values = []
        for number, row in enumerate(rows, start=1):
            cell = row[COLUMNS.index(name)]
            try:
                x = float(cell)
            except ValueError:
                raise ValueError(f"row {number}: {name} is not a number: {cell!r}") from None
            if _float_text(name, number, x) != cell:
                raise ValueError(f"row {number}: {name} {cell!r} is not repr(float), the exact spelling")
            values.append(x)
        floats[name] = np.array(values, dtype=np.float64)
    if _csv_bytes(rows) != raw:
        raise ValueError(f"{csv_path}: not in the writer's exact format (rows sorted by clip_id, minimal quoting)")

    columns = list(zip(*rows))
    return ClipScores(
        metadata=meta,
        clip_id=columns[0],
        label=columns[1],
        group_id=columns[2],
        raw_score=floats["raw_score"],
        p=floats.get("p"),
        prediction=columns[5] if "prediction" in APPLICABLE_COLUMNS[role] else None,
    )
