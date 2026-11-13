from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.audit.augmentation_corpora import audit_corpus, build_inventory, deterministic_halves
from src.audit.common import AuditError, sha256_file


def _corpus(tmp_path: Path, count=6) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    t = np.arange(800, dtype=np.float32) / 16000
    for index in range(count):
        signal = 0.1 * np.sin(2 * np.pi * (200 + index * 10) * t)
        sf.write(root / f"item-{index}.wav", signal, 16000, subtype="PCM_16")
    return root


def test_seeded_halves_are_stable_and_balanced():
    paths = [f"x/{index}.wav" for index in range(7)]
    first = deterministic_halves(paths, 2026)
    second = deterministic_halves(reversed(paths), 2026)
    assert first == second
    counts = pd.Series(first).value_counts().to_dict()
    assert counts == {"selection": 4, "training": 3}


def test_inventory_uses_only_relative_paths_and_hashes(tmp_path):
    root = _corpus(tmp_path)
    inventory = build_inventory(root)
    assert len(inventory) == 6
    assert set(inventory["half"]) == {"training", "selection"}
    assert inventory["header_ok"].all()
    assert all("/" not in path for path in inventory["rel_path"])
    assert inventory["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_audit_verifies_archive_and_license_pins(tmp_path):
    root = _corpus(tmp_path)
    archive = tmp_path / "corpus.tar"
    license_file = tmp_path / "LICENSE"
    archive.write_bytes(b"synthetic archive fixture")
    license_file.write_text("synthetic test licence", encoding="utf-8")
    report = audit_corpus(
        root=root, archive=archive, license_file=license_file,
        expected_archive_sha256=sha256_file(archive),
        expected_license_sha256=sha256_file(license_file),
        output_dir=tmp_path / "out", corpus_name="fixture",
    )
    assert report["files"] == 6
    assert report["half_counts"] == {"training": 3, "selection": 3}
    assert (tmp_path / "out" / "fixture_inventory.csv").is_file()
    assert (tmp_path / "out" / "fixture_audit.json").is_file()


def test_audit_refuses_unpinned_or_wrong_hashes(tmp_path):
    root = _corpus(tmp_path)
    archive = tmp_path / "a.tar"
    license_file = tmp_path / "LICENSE"
    archive.write_bytes(b"a")
    license_file.write_bytes(b"b")
    with pytest.raises(AuditError, match="must be pinned"):
        audit_corpus(
            root=root, archive=archive, license_file=license_file,
            expected_archive_sha256="", expected_license_sha256="",
            output_dir=tmp_path / "out", corpus_name="fixture",
        )
    with pytest.raises(AuditError, match="archive SHA-256 mismatch"):
        audit_corpus(
            root=root, archive=archive, license_file=license_file,
            expected_archive_sha256="0" * 64,
            expected_license_sha256=sha256_file(license_file),
            output_dir=tmp_path / "out2", corpus_name="fixture",
        )
