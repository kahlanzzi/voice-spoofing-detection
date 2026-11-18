"""Exact augmentation-library version guard required by D64.

The selection recipe, the materialized selection set and model-v2 training must use the same
NumPy, SciPy and SoundFile versions declared in ``requirements.txt``.  Reading the pins from the
controlling file avoids a second version list that could silently drift.
"""
from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path
from typing import Callable

PACKAGES = ("numpy", "scipy", "soundfile")
_PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s#]+)")


def requirements_path() -> Path:
    return Path(__file__).resolve().parents[2] / "requirements.txt"


def required_versions(path: Path | None = None) -> dict[str, str]:
    """Return the three exact D64 pins from requirements.txt, refusing missing/duplicate pins."""
    path = requirements_path() if path is None else Path(path)
    found: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = _PIN.match(raw.strip())
        if match is None:
            continue
        name, version = match.groups()
        canonical = name.lower().replace("_", "-")
        if canonical not in PACKAGES:
            continue
        if canonical in found:
            raise ValueError(f"{path}: duplicate exact pin for {canonical}")
        found[canonical] = version
    missing = sorted(set(PACKAGES) - set(found))
    if missing:
        raise ValueError(f"{path}: missing exact D64 pin(s): {missing}")
    return {name: found[name] for name in PACKAGES}


def require_versions(
    path: Path | None = None,
    *,
    version: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, str]:
    """Fail unless installed NumPy/SciPy/SoundFile equal the controlling requirements pins."""
    expected = required_versions(path)
    actual: dict[str, str] = {}
    mismatches: list[str] = []
    for name in PACKAGES:
        try:
            actual[name] = str(version(name))
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"D64 requires installed {name}=={expected[name]}; package is missing") from exc
        if actual[name] != expected[name]:
            mismatches.append(f"{name}: expected {expected[name]}, got {actual[name]}")
    if mismatches:
        raise ValueError("D64 augmentation library version mismatch: " + "; ".join(mismatches))
    return actual
