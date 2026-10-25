"""Verified log-Mel feature cache for the CNN: FoR train, val-A and val-B only (Stage 8, D42).

Each cached split is a folder holding <split>.npz and <split>.json:
- <split>.npz: features (n x 1 x 64 x 197 float32, the F3 log-Mel from extract_features), labels
  (n int8, 1 = spoof / FAKE, 0 = bonafide / REAL) and paths (n fixed-width unicode, sorted). It is
  written by the deterministic writer of src.models.baseline, like the MFCC cache.
- <split>.json: the MFCC cache keys plus "feature": "log_mel" and "feature_shape": [1, 64, 197].

Same guarantees as src.features.mfcc_cache, whose split, manifest and metadata helpers are reused:
the test split is refused before any file is read (D31), every sample hash is checked before
decoding, nothing is written unless every row succeeds, and loading never unpickles. The D42
train-only normalization statistics are not stored here; the training code computes them.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.mfcc_cache import (
    _COMMIT,
    _JSON_KEYS as _MFCC_JSON_KEYS,
    _NPZ_KEYS,
    _SHA256,
    CACHE_SPLITS,
    LABEL_CODES,
    SCHEMA_VERSION,
    _check_hex,
    _check_split,
    _int,
    _meta_text,
    _split_rows,
)
from src.models.baseline import _write_npz
from src.preprocessing.features import extract_features
from src.preprocessing.profile import PreprocessingProfile, load_profile
from src.preprocessing.validation import (
    F2_MANIFEST_SHA256,
    code_version,
    load_f2_manifest,
    read_verified_audio,
    sha256_file,
)

FEATURE_NAME = "log_mel"
FEATURE_SHAPE = (1, 64, 197)  # F3 CNN input (D39)
_JSON_KEYS = _MFCC_JSON_KEYS | {"feature", "feature_shape"}


def build_logmel_cache(
    root: Path,
    manifest: pd.DataFrame,
    profile: PreprocessingProfile,
    split: str,
    out_dir: Path,
    *,
    f2_manifest_sha256: str,
    profile_config_sha256: str,
    code_commit: str,
) -> dict[str, Any]:
    """Compute the (1, 64, 197) log-Mel of every row of one split and write the cache.

    The provenance values are recorded in the JSON as given; the CLI computes them from the F2
    manifest file, the profile YAML and the clean Git commit. Returns the metadata dict.
    """
    _check_split(split)  # before any file is read
    if not isinstance(profile, PreprocessingProfile):
        raise ValueError(f"profile must be a PreprocessingProfile, got {type(profile).__name__}")
    _check_hex("f2_manifest_sha256", f2_manifest_sha256, _SHA256)
    _check_hex("profile_config_sha256", profile_config_sha256, _SHA256)
    _check_hex("code_commit", code_commit, _COMMIT)
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError(f"output folder already exists: {out_dir}")
    rows = _split_rows(manifest, split)

    features = np.empty((len(rows), *FEATURE_SHAPE), dtype=np.float32)
    for i, (_, row) in enumerate(rows.iterrows()):
        waveform, sample_rate = read_verified_audio(root, row)
        log_mel = extract_features(waveform, sample_rate, profile)[1]
        if log_mel.dtype != np.float32 or log_mel.shape != FEATURE_SHAPE:
            raise ValueError(
                f"log-Mel of {row['path']} must be float32 {FEATURE_SHAPE}, "
                f"got {log_mel.dtype} {log_mel.shape}"
            )
        features[i] = log_mel
    labels = rows["label"].map(LABEL_CODES).to_numpy(dtype=np.int8)
    path_list = rows["path"].astype(str).tolist()
    paths = np.array(path_list, dtype=f"<U{max(map(len, path_list))}")

    out_dir.mkdir(parents=True)
    npz_path = out_dir / f"{split}.npz"
    try:
        _write_npz(npz_path, {"features": features, "labels": labels, "paths": paths})
        meta = {
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "feature": FEATURE_NAME,
            "feature_shape": list(FEATURE_SHAPE),
            "n_rows": int(len(rows)),
            "n_spoof": int((labels == 1).sum()),
            "n_bonafide": int((labels == 0).sum()),
            "profile_version": profile.profile_version,
            "profile_fingerprint": profile.fingerprint(),
            "profile_config_sha256": profile_config_sha256,
            "f2_manifest_sha256": f2_manifest_sha256,
            "code_commit": code_commit,
            "npz_sha256": sha256_file(npz_path),
        }
        (out_dir / f"{split}.json").write_text(_meta_text(meta), encoding="utf-8")
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return meta


def load_logmel_cache(
    out_dir: Path,
    split: str,
    profile: PreprocessingProfile,
    *,
    expected_f2_manifest_sha256: str = F2_MANIFEST_SHA256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Read and verify one cached split; return (features, labels, paths, metadata).

    Any failed check raises ValueError naming it: missing file, JSON keys or values (including the
    feature name and shape), F2 manifest hash (default: the approved D28 value), npz SHA-256,
    profile fingerprint, npz entries, shapes, dtypes or label values.
    """
    _check_split(split)
    out_dir = Path(out_dir)
    npz_path, json_path = out_dir / f"{split}.npz", out_dir / f"{split}.json"
    for path in (npz_path, json_path):
        if not path.is_file():
            raise ValueError(f"file check failed: missing cache file {path}")

    try:
        meta = json.loads(json_path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"metadata check failed: {json_path} is not valid JSON: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"metadata check failed: {json_path} must hold a JSON object")
    missing, unknown = sorted(_JSON_KEYS - set(meta)), sorted(set(meta) - _JSON_KEYS)
    if missing or unknown:
        raise ValueError(f"metadata keys check failed: missing {missing}, unknown {unknown}")
    if meta["schema_version"] != SCHEMA_VERSION or isinstance(meta["schema_version"], bool):
        raise ValueError(f"metadata check failed: unsupported schema_version {meta['schema_version']!r}")
    if meta["split"] != split:
        raise ValueError(f"metadata check failed: split is {meta['split']!r}, expected {split!r}")
    if meta["feature"] != FEATURE_NAME:
        raise ValueError(f"metadata check failed: feature is {meta['feature']!r}, expected {FEATURE_NAME!r}")
    if meta["feature_shape"] != list(FEATURE_SHAPE) or any(
        isinstance(v, bool) for v in meta["feature_shape"]
    ):
        raise ValueError(
            f"metadata check failed: feature_shape is {meta['feature_shape']!r}, "
            f"expected {list(FEATURE_SHAPE)}"
        )
    n_rows, n_spoof, n_bonafide = (_int(meta, k) for k in ("n_rows", "n_spoof", "n_bonafide"))
    if n_rows == 0 or n_spoof + n_bonafide != n_rows:
        raise ValueError("metadata check failed: n_spoof + n_bonafide must equal n_rows > 0")
    for key in ("profile_fingerprint", "profile_config_sha256", "f2_manifest_sha256", "npz_sha256"):
        _check_hex(key, meta[key], _SHA256)
    _check_hex("code_commit", meta["code_commit"], _COMMIT)
    if meta["f2_manifest_sha256"] != expected_f2_manifest_sha256:
        raise ValueError(
            f"F2 manifest check failed: cache was built from {meta['f2_manifest_sha256']}, "
            f"expected {expected_f2_manifest_sha256}"
        )

    # One read: the bytes whose hash is checked are the bytes np.load parses.
    npz_bytes = npz_path.read_bytes()
    actual = hashlib.sha256(npz_bytes).hexdigest()
    if actual != meta["npz_sha256"]:
        raise ValueError(
            f"npz SHA-256 check failed: expected {meta['npz_sha256']}, got {actual}"
        )
    if meta["profile_fingerprint"] != profile.fingerprint():
        raise ValueError(
            "profile fingerprint check failed: cache was built with "
            f"{meta['profile_fingerprint']}, profile is {profile.fingerprint()}"
        )
    if meta["profile_version"] != profile.profile_version:
        raise ValueError(
            f"profile version check failed: {meta['profile_version']!r} != {profile.profile_version!r}"
        )

    # A non-zip file makes np.load try pickle, which allow_pickle=False refuses with ValueError.
    try:
        with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as npz:
            names = set(npz.files)
            arrays = {key: npz[key] for key in names & _NPZ_KEYS}
    except (zipfile.BadZipFile, OSError, EOFError, ValueError) as exc:
        raise ValueError(f"npz read check failed: {npz_path}: {exc}") from exc
    if names != _NPZ_KEYS:
        raise ValueError(f"npz entries check failed: got {sorted(names)}")
    features, labels, paths = arrays["features"], arrays["labels"], arrays["paths"]

    if features.dtype != np.float32 or features.shape != (n_rows, *FEATURE_SHAPE):
        raise ValueError(
            f"features check failed: need float32 ({n_rows}, {', '.join(map(str, FEATURE_SHAPE))}), "
            f"got {features.dtype} {features.shape}"
        )
    if not np.isfinite(features).all():
        raise ValueError("features check failed: NaN or infinite values")
    if labels.dtype != np.int8 or labels.shape != (n_rows,):
        raise ValueError(f"labels check failed: need int8 ({n_rows},), got {labels.dtype} {labels.shape}")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels check failed: values must be 0 or 1")
    if int((labels == 1).sum()) != n_spoof or int((labels == 0).sum()) != n_bonafide:
        raise ValueError("labels check failed: class counts differ from the metadata")
    if paths.dtype.kind != "U" or paths.shape != (n_rows,):
        raise ValueError(f"paths check failed: need unicode ({n_rows},), got {paths.dtype} {paths.shape}")
    if n_rows > 1 and not (paths[:-1] < paths[1:]).all():
        raise ValueError("paths check failed: paths must be unique and sorted")
    return features, labels, paths, meta


def run(root: Path, manifest_path: Path, profile_path: Path, split: str, out: Path) -> dict[str, Any]:
    """Official build from a clean worktree: the approved F2 manifest and the F3 profile."""
    _check_split(split)
    version = code_version()  # before reading inputs or creating outputs
    profile = load_profile(Path(profile_path))
    manifest = load_f2_manifest(Path(manifest_path))
    return build_logmel_cache(
        Path(root), manifest, profile, split, Path(out),
        f2_manifest_sha256=sha256_file(Path(manifest_path)),
        profile_config_sha256=sha256_file(Path(profile_path)),
        code_commit=version["git_commit"],
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=Path, help="Local extracted FoR root")
    parser.add_argument("--manifest", required=True, type=Path, help="Approved F2 FoR manifest")
    parser.add_argument("--profile", required=True, type=Path, help="F3 preprocessing profile YAML")
    parser.add_argument("--split", required=True, choices=sorted(CACHE_SPLITS))
    parser.add_argument("--out", required=True, type=Path, help="New output folder")
    args = parser.parse_args(argv)
    meta = run(args.root, args.manifest, args.profile, args.split, args.out)
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
