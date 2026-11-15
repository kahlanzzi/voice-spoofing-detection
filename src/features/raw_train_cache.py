"""Verified raw waveform cache of the FoR train split for model v2 training (Track V; D50 (1)).

D50 augments the decoded waveform *before* the unchanged F3 profile, so the Stage 8 caches (already
past F3) cannot feed v2 training, and decoding 12,404 files from Drive every epoch would dominate
run time. This cache stores every train clip exactly as `read_verified_audio` decodes it.

One folder (outside the repository, AGENTS.md rule 7) holding:
- train_raw.npz (deterministic writer of src.models.baseline): `samples` float32 (all clips
  concatenated, as decoded), `offsets` int64 (n + 1, offsets[0] = 0, clip i is
  samples[offsets[i]:offsets[i + 1]]), `labels` int8 (1 = FAKE), `paths` unicode (sorted, unique,
  as the Stage 8 caches), `groups` unicode (F2 group IDs);
- train_raw.json: strict keys binding the arrays to the F2 manifest, the code commit and the npz.

Train split only: any other split is refused before a file is read. Every sample hash is checked
before decoding; a clip that is not 1-D or not 16,000 Hz is refused (FoR is 16 kHz mono PCM16, and
the design rests on it). Nothing is written unless every row succeeds. Loading never unpickles,
hashes the npz by streaming (the ~1.6 GB array is not held twice) and checks every array.

CLI: python -m src.features.raw_train_cache build --root R --manifest M --out O
     python -m src.features.raw_train_cache verify --dir O
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.mfcc_cache import (
    _COMMIT,
    _SHA256,
    LABEL_CODES,
    _check_hex,
    _int,
    _meta_text,
    _split_rows,
)
from src.models.baseline import _write_npz
from src.preprocessing.validation import (
    F2_MANIFEST_SHA256,
    code_version,
    load_f2_manifest,
    read_verified_audio,
    sha256_file,
)

SCHEMA_VERSION = 1
SPLIT = "train"  # the only split this cache holds (D50 (1): augmentation is train-only)
SAMPLE_RATE = 16000
NPZ_NAME = "train_raw.npz"
JSON_NAME = "train_raw.json"
NPZ_KEYS = frozenset({"samples", "offsets", "labels", "paths", "groups"})
JSON_KEYS = frozenset({
    "schema_version", "split", "n_rows", "n_spoof", "n_bonafide", "sample_rate", "total_samples",
    "f2_manifest_sha256", "code_commit", "npz_sha256",
})
REPO_ROOT = Path(__file__).resolve().parents[2]


def check_outside_repo(path: Path, what: str, repo: Path = REPO_ROOT) -> None:
    """Refuse an output path inside the repository (AGENTS.md rule 7: no audio-derived data in Git)."""
    if Path(path).resolve().is_relative_to(Path(repo).resolve()):
        raise ValueError(f"{what}: {path} is inside the repository; audio-derived data is written "
                         f"outside it (AGENTS.md rule 7)")


def _check_split(split: str) -> None:
    if split != SPLIT:
        raise ValueError(f"split must be {SPLIT!r}, got {split!r}: the raw cache holds only the "
                         f"split D50 augments; validation and held-out splits are never augmented")


def _unicode(values: list[str]) -> np.ndarray:
    return np.array(values, dtype=f"<U{max(1, max(map(len, values)))}")


def build_raw_train_cache(
    root: Path,
    manifest: pd.DataFrame,
    split: str,
    out_dir: Path,
    *,
    f2_manifest_sha256: str,
    code_commit: str,
) -> dict[str, Any]:
    """Decode every row of the train split (hash-checked) and write the cache; return the metadata."""
    _check_split(split)  # before any file is read
    _check_hex("f2_manifest_sha256", f2_manifest_sha256, _SHA256)
    _check_hex("code_commit", code_commit, _COMMIT)
    out_dir = Path(out_dir)
    check_outside_repo(out_dir, "output")
    if out_dir.exists():
        raise ValueError(f"output folder already exists: {out_dir}")
    if "group" not in manifest.columns:
        raise ValueError("manifest is missing column 'group'")
    rows = _split_rows(manifest, split)

    clips = []
    for _, row in rows.iterrows():
        waveform, sample_rate = read_verified_audio(root, row)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"{row['path']}: sample rate {sample_rate} Hz, expected {SAMPLE_RATE}")
        if waveform.ndim != 1:
            raise ValueError(f"{row['path']}: decoded shape {waveform.shape}, expected one channel")
        if waveform.dtype != np.float32:
            raise ValueError(f"{row['path']}: decoded dtype {waveform.dtype}, expected float32")
        clips.append(waveform)
    lengths = np.array([c.size for c in clips], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    samples = np.concatenate(clips).astype(np.float32, copy=False)
    del clips
    if not np.isfinite(samples).all():
        raise ValueError("decoded samples contain NaN or infinity")
    labels = rows["label"].map(LABEL_CODES).to_numpy(dtype=np.int8)
    paths = _unicode(rows["path"].astype(str).tolist())
    groups = _unicode(rows["group"].astype(str).tolist())

    out_dir.mkdir(parents=True)
    npz_path = out_dir / NPZ_NAME
    try:
        _write_npz(npz_path, {"samples": samples, "offsets": offsets, "labels": labels,
                              "paths": paths, "groups": groups})
        meta = {
            "schema_version": SCHEMA_VERSION,
            "split": SPLIT,
            "n_rows": int(len(rows)),
            "n_spoof": int((labels == 1).sum()),
            "n_bonafide": int((labels == 0).sum()),
            "sample_rate": SAMPLE_RATE,
            "total_samples": int(samples.size),
            "f2_manifest_sha256": f2_manifest_sha256,
            "code_commit": code_commit,
            "npz_sha256": sha256_file(npz_path),
        }
        (out_dir / JSON_NAME).write_text(_meta_text(meta), encoding="utf-8")
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return meta


@dataclass(frozen=True)
class RawTrainCache:
    """The verified raw train cache: clip(i) is a read-only view of clip i, as decoded."""

    samples: np.ndarray
    offsets: np.ndarray
    labels: np.ndarray
    paths: np.ndarray
    groups: np.ndarray
    meta: dict

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def clip(self, i: int) -> np.ndarray:
        n = len(self)
        if isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= int(i) < n:
            raise IndexError(f"clip index must be an int in [0, {n}), got {i!r}")
        return self.samples[self.offsets[int(i)]:self.offsets[int(i) + 1]]


def load_raw_train_cache(
    out_dir: Path,
    *,
    expected_f2_manifest_sha256: str = F2_MANIFEST_SHA256,
) -> RawTrainCache:
    """Read and verify the cache; any failed check raises ValueError naming it."""
    out_dir = Path(out_dir)
    npz_path, json_path = out_dir / NPZ_NAME, out_dir / JSON_NAME
    for path in (npz_path, json_path):
        if not path.is_file():
            raise ValueError(f"file check failed: missing cache file {path}")
    try:
        meta = json.loads(json_path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"metadata check failed: {json_path} is not valid JSON: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"metadata check failed: {json_path} must hold a JSON object")
    missing, unknown = sorted(JSON_KEYS - set(meta)), sorted(set(meta) - JSON_KEYS)
    if missing or unknown:
        raise ValueError(f"metadata keys check failed: missing {missing}, unknown {unknown}")
    n_rows, n_spoof, n_bonafide, schema, rate, total = (
        _int(meta, k) for k in ("n_rows", "n_spoof", "n_bonafide", "schema_version",
                                "sample_rate", "total_samples"))
    if schema != SCHEMA_VERSION:
        raise ValueError(f"metadata check failed: unsupported schema_version {schema!r}")
    if meta["split"] != SPLIT:
        raise ValueError(f"metadata check failed: split is {meta['split']!r}, expected {SPLIT!r}")
    if rate != SAMPLE_RATE:
        raise ValueError(f"metadata check failed: sample_rate is {rate}, expected {SAMPLE_RATE}")
    if n_rows == 0 or n_spoof + n_bonafide != n_rows:
        raise ValueError("metadata check failed: n_spoof + n_bonafide must equal n_rows > 0")
    for key in ("f2_manifest_sha256", "npz_sha256"):
        _check_hex(key, meta[key], _SHA256)
    _check_hex("code_commit", meta["code_commit"], _COMMIT)
    if meta["f2_manifest_sha256"] != expected_f2_manifest_sha256:
        raise ValueError(f"F2 manifest check failed: cache was built from "
                         f"{meta['f2_manifest_sha256']}, expected {expected_f2_manifest_sha256}")

    actual = sha256_file(npz_path)  # streaming: the large array is not read twice into memory
    if actual != meta["npz_sha256"]:
        raise ValueError(f"npz SHA-256 check failed: expected {meta['npz_sha256']}, got {actual}")
    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            names = set(npz.files)
            if names != NPZ_KEYS:
                raise ValueError(f"npz entries check failed: got {sorted(names)}")
            arrays = {key: npz[key] for key in NPZ_KEYS}
    except (zipfile.BadZipFile, OSError, EOFError) as exc:
        raise ValueError(f"npz read check failed: {npz_path}: {exc}") from exc
    samples, offsets = arrays["samples"], arrays["offsets"]
    labels, paths, groups = arrays["labels"], arrays["paths"], arrays["groups"]

    if samples.dtype != np.float32 or samples.shape != (total,):
        raise ValueError(f"samples check failed: need float32 ({total},), got {samples.dtype} "
                         f"{samples.shape}")
    if not np.isfinite(samples).all():
        raise ValueError("samples check failed: NaN or infinite values")
    if offsets.dtype != np.int64 or offsets.shape != (n_rows + 1,):
        raise ValueError(f"offsets check failed: need int64 ({n_rows + 1},), got {offsets.dtype} "
                         f"{offsets.shape}")
    if offsets[0] != 0 or offsets[-1] != total or not (np.diff(offsets) > 0).all():
        raise ValueError("offsets check failed: must start at 0, increase strictly (no empty clip) "
                         "and end at total_samples")
    if labels.dtype != np.int8 or labels.shape != (n_rows,):
        raise ValueError(f"labels check failed: need int8 ({n_rows},), got {labels.dtype} {labels.shape}")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels check failed: values must be 0 or 1")
    if int((labels == 1).sum()) != n_spoof or int((labels == 0).sum()) != n_bonafide:
        raise ValueError("labels check failed: class counts differ from the metadata")
    for name, values in (("paths", paths), ("groups", groups)):
        if values.dtype.kind != "U" or values.shape != (n_rows,):
            raise ValueError(f"{name} check failed: need unicode ({n_rows},), got {values.dtype} "
                             f"{values.shape}")
    if n_rows > 1 and not (paths[:-1] < paths[1:]).all():
        raise ValueError("paths check failed: paths must be unique and sorted")
    for array in (samples, offsets, labels, paths, groups):
        array.setflags(write=False)
    return RawTrainCache(samples=samples, offsets=offsets, labels=labels, paths=paths,
                         groups=groups, meta=meta)


def run(root: Path, manifest_path: Path, out: Path) -> dict[str, Any]:
    """Official build from a clean worktree and the approved F2 manifest."""
    version = code_version()  # before reading inputs or creating outputs
    check_outside_repo(Path(out), "output")
    manifest = load_f2_manifest(Path(manifest_path))
    return build_raw_train_cache(
        Path(root), manifest, SPLIT, Path(out),
        f2_manifest_sha256=sha256_file(Path(manifest_path)), code_commit=version["git_commit"],
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Build the raw train cache into a new folder")
    build.add_argument("--root", required=True, type=Path, help="Local extracted FoR root")
    build.add_argument("--manifest", required=True, type=Path, help="Approved F2 FoR manifest")
    build.add_argument("--out", required=True, type=Path, help="New output folder outside the repo")
    verify = sub.add_parser("verify", help="Verify an existing raw train cache")
    verify.add_argument("--dir", required=True, type=Path, help="Cache folder")
    args = parser.parse_args(argv)
    if args.command == "build":
        meta = run(args.root, args.manifest, args.out)
    else:
        meta = load_raw_train_cache(args.dir).meta
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
