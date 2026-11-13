"""V1 corpus audit and deterministic by-file split tooling (D50).

This module never downloads a corpus.  V1b supplies local archives, licence
files and extracted roots, then records the returned hashes/manifests.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .common import AUDIO_EXTS, AuditError, header_info, sha256_file


def deterministic_halves(paths: Iterable[str], seed: int = 2026) -> dict[str, str]:
    """Assign sorted unique relative paths to seeded 50/50 training/selection halves."""
    items = sorted(paths)
    if len(items) < 2 or len(items) != len(set(items)):
        raise AuditError("corpus split needs at least two unique relative paths")
    order = np.random.default_rng(seed).permutation(len(items))
    training_count = len(items) // 2
    assignment: dict[str, str] = {}
    for rank, index in enumerate(order):
        assignment[items[int(index)]] = "training" if rank < training_count else "selection"
    return assignment


def build_inventory(root: Path, seed: int = 2026) -> pd.DataFrame:
    """Hash and inspect every audio file, then assign deterministic halves."""
    root = Path(root)
    if not root.is_dir():
        raise AuditError(f"corpus root does not exist or is not a directory: {root}")
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS
    )
    if len(files) < 2:
        raise AuditError(f"fewer than two audio files found under {root}")
    relative = [path.relative_to(root).as_posix() for path in files]
    halves = deterministic_halves(relative, seed)
    rows = []
    for path, rel in zip(files, relative, strict=True):
        info = header_info(path)
        rows.append({
            "rel_path": rel,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "half": halves[rel],
            **info,
        })
    inventory = pd.DataFrame(rows).sort_values("rel_path", ignore_index=True)
    failed = inventory.loc[~inventory["header_ok"], "rel_path"].tolist()
    if failed:
        raise AuditError(f"audio header audit failed for {failed[:10]}")
    counts = inventory["half"].value_counts().to_dict()
    if abs(int(counts.get("training", 0)) - int(counts.get("selection", 0))) > 1:
        raise AuditError("corpus split is not 50/50 by file")
    return inventory


def _inventory_digest(inventory: pd.DataFrame) -> str:
    fields = inventory[["rel_path", "bytes", "sha256", "half"]].to_dict(orient="records")
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_corpus(
    *,
    root: Path,
    archive: Path,
    license_file: Path,
    expected_archive_sha256: str,
    expected_license_sha256: str,
    output_dir: Path,
    corpus_name: str,
    seed: int = 2026,
) -> dict:
    """Verify V1b pins and write a small inventory/report; never writes audio."""
    for name, value in (("archive", expected_archive_sha256), ("license", expected_license_sha256)):
        if not isinstance(value, str) or len(value) != 64:
            raise AuditError(f"expected {name} SHA-256 must be pinned before corpus use")
    archive = Path(archive)
    license_file = Path(license_file)
    if not archive.is_file() or not license_file.is_file():
        raise AuditError("archive and licence files must both exist")
    actual_archive = sha256_file(archive)
    actual_license = sha256_file(license_file)
    if actual_archive != expected_archive_sha256:
        raise AuditError(f"{corpus_name} archive SHA-256 mismatch")
    if actual_license != expected_license_sha256:
        raise AuditError(f"{corpus_name} licence SHA-256 mismatch")

    inventory = build_inventory(root, seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / f"{corpus_name.lower()}_inventory.csv"
    inventory.to_csv(inventory_path, index=False, lineterminator="\n")
    counts = {key: int(value) for key, value in inventory["half"].value_counts().items()}
    report = {
        "schema_version": 1,
        "corpus": corpus_name,
        "split_seed": seed,
        "archive_sha256": actual_archive,
        "license_sha256": actual_license,
        "files": int(len(inventory)),
        "half_counts": counts,
        "inventory_sha256": _inventory_digest(inventory),
        "inventory_file_sha256": sha256_file(inventory_path),
    }
    (output_dir / f"{corpus_name.lower()}_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
