"""Selection condition set S for model v2: folder format, writer and strict loader (D49, D50 (4)).

S is built once from val-A with seed 2026 (D50 (4)) by the V1b builder (Malak), which calls
`write_selection_set`; `load_selection_set` is the contract either way. Everything about S lives in
this module, so renaming a condition later touches only this file.

Layout of one S folder (outside the repository, AGENTS.md rule 7):
- selection_set_manifest.json: schema_version, source_split "val-A", seed 2026, conditions (the
  five below in order, each {name, parameters}), f2_manifest_sha256, profile_fingerprint,
  profile_config_sha256, augmentation_config_sha256, code_commit, runtime (object of strings),
  n_rows, files ({relative path: sha256} for every other file in the folder).
- <condition>.npz (deterministic writer): waveforms float32 (n, 32000) = the F3 waveform;
  log_mel float32 (n, 1, 64, 197) = the F3 log-Mel of that waveform (not standardized); labels int8
  (1 = FAKE); paths unicode (sorted, unique); groups unicode (F2 group IDs).
- optional sidecars (e.g. a per-clip corpus file index), allowed only if listed in `files`.

selection_set_sha256 = SHA-256 of selection_set_manifest.json, which holds every file hash.

The log-Mel stored in S is recomputed from the stored waveform for the first rows of each condition
and must match byte for byte: S must be built on the training runtime (Colab T4 2026.10, D58).
"""
from __future__ import annotations

import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.features.logmel_cache import FEATURE_SHAPE as LOGMEL_SHAPE
from src.features.mfcc_cache import LABEL_CODES
from src.features.raw_train_cache import check_outside_repo
from src.models.baseline import _write_npz
from src.preprocessing.features import log_mel_spectrogram
from src.preprocessing.profile import PreprocessingProfile
from src.preprocessing.validation import sha256_file

SCHEMA_VERSION = 1
SOURCE_SPLIT = "val-A"  # D9, D50 (4): S is built from val-A only
SEED = 2026  # D50 (4)
CONDITIONS = ("clean", "noise_10db", "reverberation", "opus_16", "narrowband")  # D50 (4), in order
MANIFEST_NAME = "selection_set_manifest.json"
WAVEFORM_SAMPLES = 32000  # F3 waveform (D39)
NPZ_KEYS = frozenset({"waveforms", "log_mel", "labels", "paths", "groups"})
FORMS = {"waveform": "waveforms", "logmel": "log_mel"}
MANIFEST_KEYS = frozenset({
    "schema_version", "source_split", "seed", "conditions", "f2_manifest_sha256",
    "profile_fingerprint", "profile_config_sha256", "augmentation_config_sha256", "code_commit",
    "runtime", "n_rows", "files",
})
# What the builder supplies; the writer adds schema_version, source_split, seed, n_rows and files.
FIELD_KEYS = frozenset({
    "conditions", "f2_manifest_sha256", "profile_fingerprint", "profile_config_sha256",
    "augmentation_config_sha256", "code_commit", "runtime",
})
EXPECTED_N_ROWS = 1330  # val-A (docs/dataset_audit.md; Stage 8 val-A cache)
EXPECTED_COUNTS = {"REAL": 660, "FAKE": 670}
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_SIDECAR = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def condition_file(name: str) -> str:
    return f"{name}.npz"


def _fail(message: str) -> None:
    raise ValueError(f"selection set: {message}")


def _require(ok: bool, message: str) -> None:
    if not ok:
        _fail(message)


def _check_fields(fields: Mapping[str, Any]) -> None:
    """The builder-supplied manifest fields: exact keys and values (shared by writer and loader)."""
    conditions = fields["conditions"]
    _require(isinstance(conditions, list) and len(conditions) == len(CONDITIONS),
             f"conditions must be a list of {len(CONDITIONS)} entries")
    for entry, name in zip(conditions, CONDITIONS):
        _require(isinstance(entry, dict) and set(entry) == {"name", "parameters"},
                 f"each condition must be {{name, parameters}}, got {entry!r}")
        _require(entry["name"] == name,
                 f"condition names and order must be {list(CONDITIONS)}, got "
                 f"{[c.get('name') if isinstance(c, dict) else c for c in conditions]}")
        _require(isinstance(entry["parameters"], dict), f"{name}.parameters must be a JSON object")
    for key in ("f2_manifest_sha256", "profile_fingerprint", "profile_config_sha256",
                "augmentation_config_sha256"):
        _require(isinstance(fields[key], str) and _HEX64.fullmatch(fields[key]) is not None,
                 f"{key} must be 64 lowercase hex, got {fields[key]!r}")
    _require(isinstance(fields["code_commit"], str) and _HEX40.fullmatch(fields["code_commit"]) is not None,
             f"code_commit must be 40 lowercase hex, got {fields['code_commit']!r}")
    runtime = fields["runtime"]
    _require(isinstance(runtime, dict) and runtime
             and all(isinstance(k, str) and isinstance(v, str) for k, v in runtime.items()),
             "runtime must be a non-empty object of strings")
    try:
        json.dumps(fields, allow_nan=False)
    except (TypeError, ValueError) as exc:
        _fail(f"manifest fields are not strict JSON: {exc}")


def _check_arrays(name: str, arrays: Mapping[str, np.ndarray], n: int | None, *,
                  keys=NPZ_KEYS) -> int:
    """Shapes, dtypes and values of one condition's arrays; return n."""
    _require(set(arrays) == set(keys), f"{name}: arrays must be {sorted(keys)}, got {sorted(arrays)}")
    labels, paths, groups = arrays["labels"], arrays["paths"], arrays["groups"]
    n = labels.shape[0] if n is None and labels.ndim == 1 else n
    _require(n is not None and n > 0, f"{name}: labels must be a non-empty vector")
    _require(labels.dtype == np.int8 and labels.shape == (n,),
             f"{name}: labels must be int8 ({n},), got {labels.dtype} {labels.shape}")
    _require(bool(np.isin(labels, (0, 1)).all()), f"{name}: labels must be 0 or 1")
    for key, values in (("paths", paths), ("groups", groups)):
        _require(values.dtype.kind == "U" and values.shape == (n,),
                 f"{name}: {key} must be unicode ({n},), got {values.dtype} {values.shape}")
    _require(n == 1 or bool((paths[:-1] < paths[1:]).all()), f"{name}: paths must be sorted and unique")
    shapes = {"waveforms": (n, WAVEFORM_SAMPLES), "log_mel": (n, *LOGMEL_SHAPE)}
    for key, shape in shapes.items():
        if key not in arrays:
            continue
        a = arrays[key]
        _require(a.dtype == np.float32 and a.shape == shape,
                 f"{name}: {key} must be float32 {shape}, got {a.dtype} {a.shape}")
        _require(bool(np.isfinite(a).all()), f"{name}: {key} contains NaN or infinite values")
    return n


def _same_rows(name: str, arrays: Mapping[str, np.ndarray], first: Mapping[str, np.ndarray]) -> None:
    for key in ("paths", "labels", "groups"):
        _require(np.array_equal(arrays[key], first[key]),
                 f"{name}: {key} differ from condition {CONDITIONS[0]!r}")


def _bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _recompute(name: str, waveforms: np.ndarray, log_mel: np.ndarray, profile, rows: int) -> None:
    """The stored log-Mel of the first `rows` rows equals log_mel_spectrogram(stored waveform)."""
    for i in range(min(rows, waveforms.shape[0])):
        again = log_mel_spectrogram(waveforms[i], profile)
        _require(_bytes_equal(again, log_mel[i]),
                 f"{name}: row {i} log-Mel recomputed from its waveform differs from the stored one "
                 f"(S must be built on the training runtime, D58)")


def _check_profile(profile) -> None:
    if not isinstance(profile, PreprocessingProfile):
        raise ValueError(f"profile must be a PreprocessingProfile, got {type(profile).__name__}")


def write_selection_set(out_dir, arrays_by_condition, manifest_fields, *, profile,
                        sidecars: Mapping[str, bytes] | None = None, recompute_rows: int = 8) -> str:
    """Validate, then write S into a new folder (manifest last); return selection_set_sha256.

    arrays_by_condition: {condition: {waveforms, log_mel, labels, paths, groups}} for the five
    CONDITIONS in order. manifest_fields: exactly FIELD_KEYS. sidecars: optional {file name: bytes},
    listed in `files`. Every check the loader makes on the content is made here first (except the
    val-A size and F2 checks, which the loader makes against its expected values); on failure the
    folder is removed.
    """
    _check_profile(profile)
    out_dir = Path(out_dir)
    check_outside_repo(out_dir, "selection set")
    if out_dir.exists():
        raise ValueError(f"selection set: output folder already exists: {out_dir}")
    _require(isinstance(manifest_fields, dict) and set(manifest_fields) == FIELD_KEYS,
             f"manifest_fields must have exactly {sorted(FIELD_KEYS)}")
    _check_fields(manifest_fields)
    _require(manifest_fields["profile_fingerprint"] == profile.fingerprint(),
             "profile_fingerprint differs from the F3 profile")
    _require(isinstance(arrays_by_condition, dict) and list(arrays_by_condition) == list(CONDITIONS),
             f"arrays_by_condition must hold {list(CONDITIONS)} in order")
    n = None
    for name in CONDITIONS:
        arrays = {k: np.asarray(v) for k, v in arrays_by_condition[name].items()}
        n = _check_arrays(name, arrays, n)
        _same_rows(name, arrays, arrays_by_condition[CONDITIONS[0]])
        _recompute(name, arrays["waveforms"], arrays["log_mel"], profile, recompute_rows)
    sidecars = dict(sidecars or {})
    reserved = {MANIFEST_NAME, *(condition_file(c) for c in CONDITIONS)}
    for file_name, data in sidecars.items():
        _require(isinstance(file_name, str) and _SIDECAR.fullmatch(file_name) is not None
                 and file_name not in reserved,
                 f"sidecar name {file_name!r} must be a plain file name, not a reserved one")
        _require(isinstance(data, bytes), f"sidecar {file_name!r} must be bytes")

    out_dir.mkdir(parents=True)
    try:
        files = {}
        for name in CONDITIONS:
            path = out_dir / condition_file(name)
            _write_npz(path, {k: np.asarray(v) for k, v in arrays_by_condition[name].items()})
            files[path.name] = sha256_file(path)
        for file_name, data in sidecars.items():
            (out_dir / file_name).write_bytes(data)
            files[file_name] = sha256_file(out_dir / file_name)
        manifest = {**manifest_fields, "schema_version": SCHEMA_VERSION,
                    "source_split": SOURCE_SPLIT, "seed": SEED, "n_rows": int(n),
                    "files": dict(sorted(files.items()))}
        text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        (out_dir / MANIFEST_NAME).write_text(text, encoding="utf-8")
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return sha256_file(out_dir / MANIFEST_NAME)


@dataclass(frozen=True)
class SelectionSet:
    """A verified S: per condition (features, labels), the common rows, the manifest and its hash."""

    form: str
    conditions: dict  # condition -> (features, labels), in CONDITIONS order
    labels: np.ndarray
    paths: np.ndarray
    groups: np.ndarray
    manifest: dict
    sha256: str

    def features(self, condition: str) -> np.ndarray:
        return self.conditions[condition][0]


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as npz:
            names = set(npz.files)
            _require(names == NPZ_KEYS, f"{path.name}: npz entries must be {sorted(NPZ_KEYS)}, "
                                        f"got {sorted(names)}")
            return {key: npz[key] for key in NPZ_KEYS}
    except (zipfile.BadZipFile, OSError, EOFError) as exc:
        raise ValueError(f"selection set: cannot read {path.name}: {exc}") from exc


def _check_f2(manifest_df, paths, labels, groups) -> None:
    """Every path is a val-A row of the F2 manifest with the same label and group."""
    rows = manifest_df.set_index("path")
    missing = [p for p in paths.tolist() if p not in rows.index]
    _require(not missing, f"{len(missing)} paths are not in the F2 manifest, first {missing[:1]}")
    found = rows.loc[paths.tolist()]
    _require(bool((found["split"] == SOURCE_SPLIT).all()), f"paths must all be {SOURCE_SPLIT} rows")
    _require(np.array_equal(found["label"].map(LABEL_CODES).to_numpy(), labels),
             "labels differ from the F2 manifest")
    _require(np.array_equal(found["group"].astype(str).to_numpy(), groups.astype(object)),
             "groups differ from the F2 manifest")


def load_selection_set(folder, profile, *, form: str, f2_manifest=None,
                       expected_n_rows: int = EXPECTED_N_ROWS,
                       expected_counts: Mapping[str, int] = EXPECTED_COUNTS,
                       clean_reference_logmel=None, recompute_rows: int = 8) -> SelectionSet:
    """Verify an S folder and return the features of `form` ("waveform" or "logmel").

    Both arrays of every condition are hash-verified and checked; only the requested one is kept,
    to keep RAM down. clean_reference_logmel: the Stage 8 val-A load_logmel_cache result
    (features, labels, paths, meta); the clean log-Mel, labels and paths must equal it byte for byte.
    f2_manifest: the F2 DataFrame; every path must be a val-A row with the same label and group.
    """
    _check_profile(profile)
    _require(form in FORMS, f"form must be one of {sorted(FORMS)}, got {form!r}")
    folder = Path(folder)
    manifest_path = folder / MANIFEST_NAME
    _require(manifest_path.is_file(), f"missing {MANIFEST_NAME} in {folder}")
    try:
        manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"selection set: {MANIFEST_NAME} is not valid JSON: {exc}") from exc
    _require(isinstance(manifest, dict), f"{MANIFEST_NAME} must hold a JSON object")
    missing = sorted(MANIFEST_KEYS - set(manifest))
    unknown = sorted(set(manifest) - MANIFEST_KEYS)
    _require(not missing and not unknown, f"manifest keys: missing {missing}, unknown {unknown}")
    _require(manifest["schema_version"] == SCHEMA_VERSION and type(manifest["schema_version"]) is int,
             f"unsupported schema_version {manifest['schema_version']!r}")
    _require(manifest["source_split"] == SOURCE_SPLIT,
             f"source_split must be {SOURCE_SPLIT!r} (D9, D50 (4)), got {manifest['source_split']!r}")
    _require(manifest["seed"] == SEED and type(manifest["seed"]) is int,
             f"seed must be {SEED} (D50 (4)), got {manifest['seed']!r}")
    _check_fields({k: manifest[k] for k in FIELD_KEYS})
    _require(manifest["profile_fingerprint"] == profile.fingerprint(),
             f"profile_fingerprint {manifest['profile_fingerprint']} differs from the F3 profile "
             f"{profile.fingerprint()}")
    _require(type(manifest["n_rows"]) is int and manifest["n_rows"] == expected_n_rows,
             f"n_rows must be {expected_n_rows}, got {manifest['n_rows']!r}")

    files = manifest["files"]
    _require(isinstance(files, dict), "files must be an object {relative path: sha256}")
    present = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*")
                     if p.is_file() and p.name != MANIFEST_NAME)
    _require(sorted(files) == present,
             f"files listed {sorted(files)} differ from files present {present}")
    for name in CONDITIONS:
        _require(condition_file(name) in files, f"missing {condition_file(name)} in files")
    for rel, digest in files.items():
        _require(isinstance(digest, str) and _HEX64.fullmatch(digest) is not None,
                 f"files[{rel!r}] must be 64 lowercase hex")
        actual = sha256_file(folder / rel)
        _require(actual == digest, f"{rel}: SHA-256 {actual} differs from the manifest {digest}")

    keep = FORMS[form]
    conditions, first = {}, None
    for name in CONDITIONS:
        arrays = _read_npz(folder / condition_file(name))
        _check_arrays(name, arrays, expected_n_rows)
        if first is None:
            first = {k: arrays[k] for k in ("paths", "labels", "groups")}
        _same_rows(name, arrays, first)
        _recompute(name, arrays["waveforms"], arrays["log_mel"], profile, recompute_rows)
        if name == CONDITIONS[0] and clean_reference_logmel is not None:
            ref_features, ref_labels, ref_paths = clean_reference_logmel[:3]
            _require(_bytes_equal(np.asarray(ref_paths), arrays["paths"])
                     and _bytes_equal(np.asarray(ref_labels), arrays["labels"]),
                     "clean paths or labels differ from the Stage 8 val-A cache")
            _require(_bytes_equal(np.asarray(ref_features), arrays["log_mel"]),
                     "clean log-Mel differs from the Stage 8 val-A cache")
        conditions[name] = (arrays[keep], first["labels"])
        del arrays

    labels = first["labels"]
    counts = {"REAL": int((labels == 0).sum()), "FAKE": int((labels == 1).sum())}
    _require(counts == dict(expected_counts), f"label counts {counts} differ from {dict(expected_counts)}")
    if f2_manifest is not None:
        _check_f2(f2_manifest, first["paths"], labels, first["groups"])
    return SelectionSet(form=form, conditions=conditions, labels=labels, paths=first["paths"],
                        groups=first["groups"], manifest=manifest,
                        sha256=sha256_file(manifest_path))
