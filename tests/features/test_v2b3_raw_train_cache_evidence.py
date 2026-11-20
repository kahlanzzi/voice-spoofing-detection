"""The committed V2b-3 raw train cache evidence (docs/audit_evidence/v2b3_raw_train_cache/). Torch-free."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[2] / "docs" / "audit_evidence" / "v2b3_raw_train_cache"
JSON_SHA256 = "34065507847689817d9b29af6972108d9c2fc8d42edc85cbad3908c8bc1d25a9"
NPZ_SHA256 = "026a917debfd262ff3e507ea073fa33ef14280fd8dbe235366906f549b4a75e8"


def test_evidence_file_is_the_supplied_one():
    assert hashlib.sha256((EVIDENCE / "train_raw.json").read_bytes()).hexdigest() == JSON_SHA256


def test_no_all_zero_clip_and_the_recorded_counts():
    meta = json.loads((EVIDENCE / "train_raw.json").read_text(encoding="utf-8"))
    assert meta["n_all_zero_clips"] == 0  # D62 not needed
    assert (meta["split"], meta["n_rows"], meta["n_spoof"], meta["n_bonafide"]) == ("train", 12404, 6244, 6160)
    assert meta["total_samples"] == 12404 * 32000 and meta["sample_rate"] == 16000
    assert meta["npz_sha256"] == NPZ_SHA256
    assert meta["code_commit"] == "37b236b936afffd150bde5d9ad9020fb68952f2d"
    assert meta["f2_manifest_sha256"] == "badfe948fb7b833357783dd18be00f6bb36253cf8699dc4c4beb4df2b07706f8"


def test_readme_states_the_hashes():
    text = (EVIDENCE / "README.md").read_text(encoding="utf-8")
    assert JSON_SHA256 in text and NPZ_SHA256 in text and "D62 is not needed" in text
