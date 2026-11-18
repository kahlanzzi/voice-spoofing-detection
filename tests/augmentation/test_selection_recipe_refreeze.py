import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import src.augmentation.selection_recipe_refreeze as refreeze
from src.augmentation.selection_recipe_refreeze import (
    compare_recipes,
    load_committed_old,
    make_new_report,
)
from src.augmentation.selection_set_recipe import CONDITIONS, RECIPE_COLUMNS


def _recipes():
    rows = []
    for condition in CONDITIONS:
        resource = "noise.wav" if condition == "recorded_noise_10db" else (
            "rir.wav" if condition == "reverberation" else ""
        )
        rows.append({
            "source_path": "FAKE/one.wav",
            "label": "FAKE",
            "subgroup": "FAKE_other",
            "group": "1",
            "source_file_sha256": "1" * 64,
            "condition": condition,
            "condition_seed": "2026",
            "resource_path": resource,
            "resource_file_sha256": "2" * 64 if resource else "",
            "resource_channel": "0" if resource else "",
            "parameters_json": "{}",
            "sample_rate_hz": "16000",
            "samples": "32000",
            "waveform_float32_sha256": "a" * 64,
        })
    old = pd.DataFrame(rows, columns=RECIPE_COLUMNS)
    new = old.copy()
    new.loc[new["condition"] == "reverberation", "waveform_float32_sha256"] = "b" * 64
    return old, new


def _inventories():
    return (
        [{"rel_path": "noise.wav", "half": "selection"}],
        [{"rel_path": "rir.wav", "half": "selection"}],
    )


def test_comparison_accepts_only_hash_changes_and_lists_old_and_new():
    old, new = _recipes()
    noise, rirs = _inventories()
    result = compare_recipes(old, new, noise_rows=noise, rir_rows=rirs)
    assert result["result"] == "passed"
    assert result["row_count"] == 5
    assert result["changed_waveform_hashes_by_condition"] == {"reverberation": 1}
    assert result["changed_waveform_hashes"][0]["old_waveform_float32_sha256"] == "a" * 64
    assert result["changed_waveform_hashes"][0]["new_waveform_float32_sha256"] == "b" * 64
    assert result["resource_halves"] == {
        "recorded_noise_10db": "selection",
        "reverberation": "selection",
    }


@pytest.mark.parametrize("column", [
    "source_path", "condition_seed", "resource_path", "resource_channel", "parameters_json",
])
def test_comparison_refuses_invariant_changes(column):
    old, new = _recipes()
    noise, rirs = _inventories()
    new.loc[0, column] = str(new.loc[0, column]) + "changed"
    with pytest.raises(ValueError, match="changed|row set"):
        compare_recipes(old, new, noise_rows=noise, rir_rows=rirs)


def test_comparison_refuses_nonselection_resource_half():
    old, new = _recipes()
    noise, rirs = _inventories()
    noise[0]["half"] = "training"
    with pytest.raises(ValueError, match="selection half"):
        compare_recipes(old, new, noise_rows=noise, rir_rows=rirs)


def test_committed_old_is_bound_to_report_file_and_semantic_hashes(tmp_path):
    manifest_dir = tmp_path / "data" / "manifests" / "augmentation"
    manifest_dir.mkdir(parents=True)
    recipe_path = manifest_dir / "selection_set_s.csv"
    report_path = manifest_dir / "selection_set_s_report.json"
    old, _ = _recipes()
    old.to_csv(recipe_path, index=False, lineterminator="\n")
    report = {
        "manifest_file_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
        "manifest_semantic_sha256": refreeze._semantic_sha256(old),
    }
    report_bytes = (json.dumps(report, indent=1) + "\n\n").encode("utf-8")
    report_path.write_bytes(report_bytes)
    loaded_path, loaded_report_path, loaded, loaded_report = load_committed_old(tmp_path)
    assert loaded_path == recipe_path
    assert loaded_report_path == report_path
    assert loaded_report_path.read_bytes() == report_bytes
    assert loaded.equals(old)
    assert loaded_report == report
    report["manifest_file_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from report"):
        load_committed_old(tmp_path)


def test_inventory_hashes_are_bound_to_committed_csv_bytes(tmp_path):
    manifest_dir = tmp_path / "data" / "manifests" / "augmentation"
    manifest_dir.mkdir(parents=True)
    musan = manifest_dir / "musan_inventory.csv"
    rirs = manifest_dir / "rirs_noises_inventory.csv"
    musan.write_bytes(b"rel_path,half\nnoise.wav,selection\n")
    rirs.write_bytes(b"rel_path,half\nrir.wav,selection\n")
    report = {
        "musan_inventory_sha256": hashlib.sha256(musan.read_bytes()).hexdigest(),
        "rirs_inventory_sha256": hashlib.sha256(rirs.read_bytes()).hexdigest(),
    }
    refreeze._validate_inventory_hashes(manifest_dir, report)

    musan.write_bytes(musan.read_bytes() + b"tampered.wav,selection\n")
    with pytest.raises(ValueError, match="musan_inventory.csv.*differs from report"):
        refreeze._validate_inventory_hashes(manifest_dir, report)


def test_new_report_replaces_old_provenance_with_d64_context():
    report = make_new_report(
        {"repo_base_commit": "old", "decisions": ["D49"], "other": "retained"},
        recipe_sha256="a" * 64,
        semantic_sha256="b" * 64,
        commit="c" * 40,
        versions={"numpy": "2.5.3", "scipy": "1.18.1", "soundfile": "0.14.0"},
        runtime={"gpu": "Tesla T4", "colab_runtime_version": "2026.10"},
    )
    assert report["repo_base_commit"] == "c" * 40
    assert report["d64_refreeze_commit"] == "c" * 40
    assert report["decisions"] == ["D49", "D50", "D56", "D64"]
    assert report["library_versions"]["scipy"] == "1.18.1"
    assert report["runtime"]["gpu"] == "Tesla T4"
    assert report["other"] == "retained"


def test_regenerate_checks_runtime_and_all_invariants_before_creating_out(tmp_path, monkeypatch):
    old_path = tmp_path / "old.csv"
    pd.DataFrame(columns=RECIPE_COLUMNS).to_csv(old_path, index=False)
    monkeypatch.setattr(refreeze, "require_versions", lambda: {"scipy": "1.18.1"})
    monkeypatch.setattr(refreeze, "code_version", lambda: {"git_commit": "c" * 40})
    monkeypatch.setattr(
        refreeze,
        "load_committed_old",
        lambda _repo: (old_path, tmp_path / "old-report.json", pd.DataFrame(columns=RECIPE_COLUMNS), {}),
    )
    monkeypatch.setattr(
        refreeze,
        "load_augmentation_config",
        lambda _path: SimpleNamespace(
            corpus_pins_complete=True,
            ffmpeg_pin_complete=True,
            ffmpeg={},
            sample_rate=16000,
            codec_options=[],
        ),
    )
    monkeypatch.setattr(
        refreeze,
        "FFmpegCodecBackend",
        lambda *_args: SimpleNamespace(verify_pin=lambda: None),
    )
    calls = []
    monkeypatch.setattr(
        refreeze,
        "official_runtime_record",
        lambda _codec, runtime: calls.append(runtime) or {"colab_runtime_version": runtime},
    )
    monkeypatch.setattr(refreeze, "_validate_inventory_hashes", lambda *_args: None)
    monkeypatch.setattr(refreeze, "_records", lambda *_args: [])
    monkeypatch.setattr(
        refreeze,
        "load_f2_manifest",
        lambda _path: pd.DataFrame(columns=["split", "path"]),
    )
    monkeypatch.setattr(
        refreeze,
        "compare_recipes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("comparison failed")),
    )
    out = tmp_path / "outside" / "out"
    with pytest.raises(ValueError, match="comparison failed"):
        refreeze.regenerate(
            tmp_path,
            tmp_path / "f2.csv",
            tmp_path,
            tmp_path,
            out,
            "2026.10",
        )
    assert calls == ["2026.10"]
    assert not out.exists()


def test_committed_d64_evidence_changes_only_reverberation_hashes():
    repo = Path(__file__).resolve().parents[2]
    evidence = repo / "docs" / "audit_evidence" / "trackV_v1c_d64"
    comparison = json.loads(
        (evidence / "selection_recipe_refreeze_comparison.json").read_text(encoding="utf-8")
    )

    assert comparison["result"] == "passed"
    assert comparison["code_commit"] == "41ad8bc8b95ef8d9d178e8b6986fcf9a192e48aa"
    assert comparison["row_count"] == 6650
    assert comparison["source_count"] == 1330
    assert comparison["changed_waveform_hash_count"] == 1330
    assert comparison["unchanged_waveform_hash_count"] == 5320
    assert comparison["changed_waveform_hashes_by_condition"] == {"reverberation": 1330}
    assert all(comparison[name] for name in (
        "rows_identical", "sources_identical", "conditions_identical", "parameters_identical",
        "seeds_identical", "resources_identical", "resource_halves_identical",
    ))
    assert comparison["runtime"]["colab_runtime_version"] == "2026.10"
    assert comparison["runtime"]["gpu"] == "Tesla T4"
