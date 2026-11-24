"""Torch-free integrity checks for the official Track V V1c real-build evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs" / "audit_evidence" / "trackV_v1c_build"
MANIFEST_SHA256 = "7aaa69f916027f52bfa29a8cce935f48d3a9ac7d9481cf8a0fb04fdebfdc092e"
RESOURCES_SHA256 = "0afc6e439d3201fd26dae921db62d082ae94e4dd214cbb404e19526f6f11aa24"
REPORT_SHA256 = "dfd53330cc6521c377d83b46b09e6cc6573548cccbe9f496c7eeb14b3578ffff"
RESOURCE_NPZ_SHA256 = "becc00609a667a68eda02a5184466e42cdb9a232912607639ee69e67f2e393c8"
RUN_COMMIT = "d5ffe18da5912b4a4d4174f738dbe1b2d5603fcd"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def test_evidence_files_are_the_supplied_bytes():
    assert _sha(EVIDENCE / "selection_set_manifest.json") == MANIFEST_SHA256
    assert _sha(EVIDENCE / "training_resources.json") == RESOURCES_SHA256
    assert _sha(EVIDENCE / "v1c_build_report.json") == REPORT_SHA256


def test_build_report_binds_both_outputs_and_official_runtime():
    report = _json("v1c_build_report.json")
    assert report["code_commit"] == RUN_COMMIT
    assert report["built_in_this_session"] == {
        "resource_cache": True,
        "selection_set": True,
        "stage8_logmel": {"train": False, "val-A": False},
    }
    assert report["selection_set_sha256"] == MANIFEST_SHA256
    assert report["resource_cache_json_sha256"] == RESOURCES_SHA256
    assert report["resource_cache_npz_sha256"] == RESOURCE_NPZ_SHA256
    assert report["runtime"] == {
        "colab_runtime_version": "2026.10",
        "ffmpeg_version_line": (
            "ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023 the FFmpeg developers"
        ),
        "gpu_name": "Tesla T4",
        "python": "3.13.16",
        "torch": "2.11.0+cu130",
    }
    assert report["require_versions"] == {
        "numpy": "2.5.3", "scipy": "1.18.1", "soundfile": "0.14.0"
    }


def test_selection_manifest_records_all_five_conditions():
    manifest = _json("selection_set_manifest.json")
    assert manifest["code_commit"] == RUN_COMMIT
    assert (manifest["source_split"], manifest["n_rows"], manifest["seed"]) == (
        "val-A", 1330, 2026
    )
    assert [item["name"] for item in manifest["conditions"]] == [
        "clean", "recorded_noise_10db", "reverberation", "opus_16", "narrowband_mulaw"
    ]
    assert set(manifest["files"]) == {
        "clean.npz", "recorded_noise_10db.npz", "reverberation.npz",
        "opus_16.npz", "narrowband_mulaw.npz",
    }
    assert all(len(digest) == 64 for digest in manifest["files"].values())
    assert manifest["runtime"]["gpu"] == "Tesla T4"
    assert manifest["runtime"]["colab_runtime_version"] == "2026.10"


def test_resource_cache_matches_committed_inventories_and_zero_gate():
    resources = _json("training_resources.json")
    report = _json("v1c_build_report.json")
    inventory = ROOT / "data" / "manifests" / "augmentation"

    assert resources["code_commit"] == RUN_COMMIT
    assert (resources["noise_count"], resources["rir_count"]) == (465, 30162)
    assert (resources["n_all_zero_noise"], resources["n_all_zero_rir"]) == (0, 0)
    assert (resources["sample_rate"], resources["split_seed"]) == (16000, 2026)
    assert resources["npz_sha256"] == RESOURCE_NPZ_SHA256
    assert resources["noise_inventory_sha256"] == _sha(inventory / "musan_inventory.csv")
    assert resources["rir_inventory_sha256"] == _sha(inventory / "rirs_noises_inventory.csv")
    for key in (
        "code_commit", "noise_archive_sha256", "noise_count", "noise_inventory_sha256",
        "rir_archive_sha256", "rir_count", "rir_inventory_sha256",
    ):
        assert report["resource_cache"][key] == resources[key]
