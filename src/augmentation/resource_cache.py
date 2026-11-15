"""Verified training-half MUSAN/RIRS resource cache for Track V augmentation (D50, D56).

The cache is built outside the repository and contains decoded 16 kHz resources only.  Every source
file is checked against the committed V1b inventory before decoding.  Selection-half rows are never
accepted, and all-zero resources make the whole build fail without leaving a partial cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

from src.augmentation.config import load_augmentation_config
from src.augmentation.pipeline import AugmentationResources
from src.features.raw_train_cache import check_outside_repo
from src.models.baseline import _write_npz
from src.preprocessing.validation import code_version, sha256_file

SCHEMA_VERSION = 1
SAMPLE_RATE = 16000
SEED = 2026
NPZ_NAME = "training_resources.npz"
JSON_NAME = "training_resources.json"
NOISE_COUNT = 465
RIR_COUNT = 30162
NPZ_KEYS = frozenset({
    "noise_samples", "noise_offsets", "noise_paths", "noise_sha256",
    "rir_samples", "rir_offsets", "rir_paths", "rir_sha256", "rir_channels",
})
JSON_KEYS = frozenset({
    "schema_version", "sample_rate", "split_seed", "noise_count", "rir_count",
    "noise_inventory_sha256", "rir_inventory_sha256", "noise_archive_sha256",
    "rir_archive_sha256", "n_all_zero_noise", "n_all_zero_rir", "code_commit",
    "npz_sha256",
})


def _hex(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be 64 lowercase hex characters")
    return value


def _unicode(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=f"<U{max(1, max(map(len, values)))}")


def _training_rows(inventory: pd.DataFrame, name: str, expected: int) -> pd.DataFrame:
    required = {"rel_path", "sha256", "half"}
    missing = sorted(required - set(inventory.columns))
    if missing:
        raise ValueError(f"{name} inventory is missing columns {missing}")
    rows = inventory.loc[inventory["half"] == "training"].sort_values("rel_path", kind="stable")
    if len(rows) != expected:
        raise ValueError(f"{name} training half must contain {expected} files, found {len(rows)}")
    if len(rows) != rows["rel_path"].nunique():
        raise ValueError(f"{name} training paths must be unique")
    return rows.reset_index(drop=True)


def training_rir_channel(rel_path: str, channels: int) -> int:
    """D50 V1c channel rule: mono channel 0, otherwise seeded RandomState choice."""
    if channels < 1:
        raise ValueError(f"RIR {rel_path!r} has no channels")
    if channels == 1:
        return 0
    payload = f"{SEED}\0{rel_path}\0training_channel".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    return int(np.random.RandomState(seed).randint(0, channels))


def _decode(root: Path, row: pd.Series, name: str) -> tuple[np.ndarray, int]:
    rel = str(row["rel_path"])
    path = Path(root) / rel
    expected, actual = str(row["sha256"]), sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} {rel}: SHA-256 {actual} differs from inventory {expected}")
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{name} {rel}: sample rate {rate}, expected {SAMPLE_RATE}")
    if data.shape[0] == 0 or data.shape[1] == 0 or not np.isfinite(data).all():
        raise ValueError(f"{name} {rel}: decoded resource is empty or non-finite")
    return np.asarray(data, dtype=np.float32), int(rate)


def _packed(waveforms: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.concatenate(([0], np.cumsum([x.size for x in waveforms]))).astype(np.int64)
    samples = np.concatenate(waveforms).astype(np.float32, copy=False)
    return samples, offsets


def build_training_resources(
    noise_root: Path,
    rirs_root: Path,
    noise_inventory: pd.DataFrame,
    rir_inventory: pd.DataFrame,
    out_dir: Path,
    *,
    noise_inventory_sha256: str,
    rir_inventory_sha256: str,
    noise_archive_sha256: str,
    rir_archive_sha256: str,
    code_commit: str,
    expected_noise_count: int = NOISE_COUNT,
    expected_rir_count: int = RIR_COUNT,
) -> dict[str, Any]:
    """Hash-check, decode, channel-select and write the training-half resource cache."""
    for name, value in (
        ("noise_inventory_sha256", noise_inventory_sha256),
        ("rir_inventory_sha256", rir_inventory_sha256),
        ("noise_archive_sha256", noise_archive_sha256),
        ("rir_archive_sha256", rir_archive_sha256),
    ):
        _hex(name, value)
    if not isinstance(code_commit, str) or len(code_commit) != 40 or any(
        c not in "0123456789abcdef" for c in code_commit
    ):
        raise ValueError("code_commit must be 40 lowercase hex characters")
    out_dir = Path(out_dir)
    check_outside_repo(out_dir, "training resource cache")
    if out_dir.exists():
        raise ValueError(f"training resource cache already exists: {out_dir}")

    noise_rows = _training_rows(noise_inventory, "MUSAN", expected_noise_count)
    rir_rows = _training_rows(rir_inventory, "RIRS_NOISES", expected_rir_count)
    noise_waveforms: list[np.ndarray] = []
    rir_waveforms: list[np.ndarray] = []
    rir_channels: list[int] = []
    zero_noise: list[str] = []
    zero_rir: list[str] = []
    for _, row in noise_rows.iterrows():
        data, _ = _decode(noise_root, row, "MUSAN")
        if data.shape[1] != 1:
            raise ValueError(f"MUSAN {row['rel_path']}: expected mono, got {data.shape[1]} channels")
        waveform = data[:, 0]
        if not np.any(waveform):
            zero_noise.append(str(row["rel_path"]))
        noise_waveforms.append(waveform)
    for _, row in rir_rows.iterrows():
        data, _ = _decode(rirs_root, row, "RIRS_NOISES")
        channel = training_rir_channel(str(row["rel_path"]), data.shape[1])
        waveform = data[:, channel]
        if not np.any(waveform):
            zero_rir.append(str(row["rel_path"]))
        rir_waveforms.append(waveform)
        rir_channels.append(channel)
    if zero_noise or zero_rir:
        paths = [*(f"MUSAN: {p}" for p in zero_noise), *(f"RIRS_NOISES: {p}" for p in zero_rir)]
        raise ValueError("all-zero training resources are refused:\n" + "\n".join(paths))

    noise_samples, noise_offsets = _packed(noise_waveforms)
    rir_samples, rir_offsets = _packed(rir_waveforms)
    arrays = {
        "noise_samples": noise_samples,
        "noise_offsets": noise_offsets,
        "noise_paths": _unicode(noise_rows["rel_path"].astype(str).tolist()),
        "noise_sha256": _unicode(noise_rows["sha256"].astype(str).tolist()),
        "rir_samples": rir_samples,
        "rir_offsets": rir_offsets,
        "rir_paths": _unicode(rir_rows["rel_path"].astype(str).tolist()),
        "rir_sha256": _unicode(rir_rows["sha256"].astype(str).tolist()),
        "rir_channels": np.asarray(rir_channels, dtype=np.int32),
    }
    out_dir.mkdir(parents=True)
    try:
        npz_path = out_dir / NPZ_NAME
        _write_npz(npz_path, arrays)
        meta = {
            "schema_version": SCHEMA_VERSION, "sample_rate": SAMPLE_RATE, "split_seed": SEED,
            "noise_count": len(noise_rows), "rir_count": len(rir_rows),
            "noise_inventory_sha256": noise_inventory_sha256,
            "rir_inventory_sha256": rir_inventory_sha256,
            "noise_archive_sha256": noise_archive_sha256,
            "rir_archive_sha256": rir_archive_sha256,
            "n_all_zero_noise": 0, "n_all_zero_rir": 0,
            "code_commit": code_commit, "npz_sha256": sha256_file(npz_path),
        }
        (out_dir / JSON_NAME).write_text(
            json.dumps(meta, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return meta


def _views(samples: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, ...]:
    result = tuple(samples[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1))
    for item in result:
        item.setflags(write=False)
    return result


def load_training_resources(
    directory: Path,
    *,
    expected_noise_count: int = NOISE_COUNT,
    expected_rir_count: int = RIR_COUNT,
) -> tuple[AugmentationResources, dict[str, Any]]:
    """Strictly verify a cache and return read-only waveform views plus its metadata."""
    directory = Path(directory)
    npz_path, meta_path = directory / NPZ_NAME, directory / JSON_NAME
    if not npz_path.is_file() or not meta_path.is_file():
        raise ValueError(f"training resource cache files are missing from {directory}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"training resource metadata is invalid: {exc}") from exc
    if not isinstance(meta, dict) or set(meta) != JSON_KEYS:
        raise ValueError("training resource metadata keys differ from the V1c contract")
    if meta["schema_version"] != SCHEMA_VERSION or meta["sample_rate"] != SAMPLE_RATE \
            or meta["split_seed"] != SEED:
        raise ValueError("training resource metadata version/rate/seed check failed")
    if meta["noise_count"] != expected_noise_count or meta["rir_count"] != expected_rir_count:
        raise ValueError(
            "training resource counts differ from the audited halves: "
            f"{meta['noise_count']}/{meta['rir_count']} != "
            f"{expected_noise_count}/{expected_rir_count}"
        )
    for key in ("noise_inventory_sha256", "rir_inventory_sha256", "noise_archive_sha256",
                "rir_archive_sha256", "npz_sha256"):
        _hex(key, meta[key])
    if not isinstance(meta["code_commit"], str) or len(meta["code_commit"]) != 40 or any(
        c not in "0123456789abcdef" for c in meta["code_commit"]
    ):
        raise ValueError("training resource code_commit must be 40 lowercase hex characters")
    if meta["n_all_zero_noise"] != 0 or meta["n_all_zero_rir"] != 0:
        raise ValueError("training resource metadata records all-zero resources")
    actual = sha256_file(npz_path)
    if actual != meta["npz_sha256"]:
        raise ValueError(f"training resource npz SHA-256 mismatch: {actual} != {meta['npz_sha256']}")
    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            if set(npz.files) != NPZ_KEYS:
                raise ValueError(f"training resource arrays differ: {sorted(npz.files)}")
            arrays = {name: npz[name] for name in NPZ_KEYS}
    except (OSError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"cannot read training resource npz: {exc}") from exc
    counts = {"noise": int(meta["noise_count"]), "rir": int(meta["rir_count"])}
    for prefix in ("noise", "rir"):
        n = counts[prefix]
        samples, offsets = arrays[f"{prefix}_samples"], arrays[f"{prefix}_offsets"]
        paths, hashes = arrays[f"{prefix}_paths"], arrays[f"{prefix}_sha256"]
        if samples.dtype != np.float32 or samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError(f"{prefix}_samples check failed")
        if offsets.dtype != np.int64 or offsets.shape != (n + 1,) or offsets[0] != 0 \
                or offsets[-1] != samples.size or not (np.diff(offsets) > 0).all():
            raise ValueError(f"{prefix}_offsets check failed")
        if paths.dtype.kind != "U" or paths.shape != (n,) or (n > 1 and not (paths[:-1] < paths[1:]).all()):
            raise ValueError(f"{prefix}_paths check failed")
        if hashes.dtype.kind != "U" or hashes.shape != (n,) or any(len(x) != 64 for x in hashes):
            raise ValueError(f"{prefix}_sha256 check failed")
        if any(not np.any(samples[offsets[i]:offsets[i + 1]]) for i in range(n)):
            raise ValueError(f"{prefix} cache contains an all-zero resource")
    channels = arrays["rir_channels"]
    if channels.dtype != np.int32 or channels.shape != (counts["rir"],) or (channels < 0).any():
        raise ValueError("rir_channels check failed")
    for array in arrays.values():
        array.setflags(write=False)
    resources = AugmentationResources(
        noise_waveforms=_views(arrays["noise_samples"], arrays["noise_offsets"]),
        rir_waveforms=_views(arrays["rir_samples"], arrays["rir_offsets"]),
    )
    return resources, meta


def run(noise_root: Path, rirs_root: Path, out: Path) -> dict[str, Any]:
    """Official build using committed V1b inventories and pins from a clean worktree."""
    version = code_version()
    repo = Path(__file__).resolve().parents[2]
    manifest_dir = repo / "data" / "manifests" / "augmentation"
    noise_path, rir_path = manifest_dir / "musan_inventory.csv", manifest_dir / "rirs_noises_inventory.csv"
    config = load_augmentation_config(repo / "configs" / "augmentation.yaml")
    return build_training_resources(
        noise_root, rirs_root, pd.read_csv(noise_path), pd.read_csv(rir_path), out,
        noise_inventory_sha256=sha256_file(noise_path), rir_inventory_sha256=sha256_file(rir_path),
        noise_archive_sha256=str(config.corpora["noise"]["archive_sha256"]),
        rir_archive_sha256=str(config.corpora["rir"]["archive_sha256"]),
        code_commit=version["git_commit"],
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--musan-noise-root", required=True, type=Path)
    build.add_argument("--rirs-root", required=True, type=Path)
    build.add_argument("--out", required=True, type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        run(args.musan_noise_root, args.rirs_root, args.out)
        _, meta = load_training_resources(args.out)
    else:
        _, meta = load_training_resources(args.dir)
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
