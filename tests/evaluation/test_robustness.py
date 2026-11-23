"""Stage 14 condition/config tests; synthetic waveforms only, no datasets or model scoring."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import src.evaluation.stage14_run as stage14_module
from src.evaluation.clip_scores import ClipScores, write_clip_scores
from src.evaluation.run_registry import RunRegistry
from src.evaluation.robustness import (
    CONDITION_IDS,
    RobustnessError,
    apply_condition,
    compare_condition,
    load_robustness_config,
)
from src.evaluation.stage14_run import read_stage11_clean_scores
from src.evaluation.stage11_run import MODEL_IDS


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "robustness.yaml"


class FakeCodec:
    def __init__(self):
        self.calls = []

    def round_trip(self, waveform, option):
        self.calls.append(option)
        return np.asarray(waveform * np.float32(0.5), dtype=np.float32)


@pytest.fixture
def config():
    return load_robustness_config(CONFIG)


@pytest.fixture
def waveform():
    time = np.arange(32_000, dtype=np.float64) / 16_000
    return np.asarray(0.25 * np.sin(2 * np.pi * 440 * time), dtype=np.float32)


def test_config_loads_the_complete_frozen_grid(config):
    assert config.sample_rate == 16_000
    assert CONDITION_IDS == (
        "white_gaussian_noise_30db", "white_gaussian_noise_20db", "white_gaussian_noise_10db",
        "mp3_128kbps", "mp3_64kbps", "mp3_32kbps",
        "low_pass_6000hz", "low_pass_4000hz", "low_pass_3400hz",
    )
    assert config.bootstrap == {
        "unit": "d14_group", "n_resamples": 2000, "seed": 2026, "confidence": 0.95}


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda value: value["conditions"]["mp3"]["bitrate_kbps"].reverse(), "MP3"),
        (lambda value: value["conditions"]["low_pass"].update({"order": 6}), "low-pass"),
        (lambda value: value["bootstrap"].update({"seed": 1}), "bootstrap"),
        (lambda value: value.update({"unknown": 1}), "unknown keys"),
    ],
)
def test_config_rejects_drift(tmp_path, edit, match):
    value = copy.deepcopy(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    edit(value)
    path = tmp_path / "robustness.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(RobustnessError, match=match):
        load_robustness_config(path)


@pytest.mark.parametrize("condition", CONDITION_IDS[:3])
def test_white_noise_is_per_clip_deterministic_and_has_requested_snr(config, waveform, condition):
    first = apply_condition(waveform, condition_id=condition, clip_id="REAL/a.wav", config=config)
    second = apply_condition(waveform, condition_id=condition, clip_id="REAL/a.wav", config=config)
    other = apply_condition(waveform, condition_id=condition, clip_id="REAL/b.wav", config=config)
    assert first.tobytes() == second.tobytes()
    assert first.tobytes() != other.tobytes()
    noise = first.astype(np.float64) - waveform.astype(np.float64)
    actual = 10 * np.log10(np.mean(waveform.astype(np.float64) ** 2) / np.mean(noise ** 2))
    assert actual == pytest.approx(int(condition.split("_")[-1][:-2]), abs=2e-4)


@pytest.mark.parametrize("condition,bitrate", tuple(zip(CONDITION_IDS[3:6], (128, 64, 32))))
def test_mp3_uses_the_injected_pinned_backend(config, waveform, condition, bitrate):
    codec = FakeCodec()
    result = apply_condition(waveform, condition_id=condition, clip_id="REAL/a.wav", config=config,
                             codec_backend=codec)
    assert codec.calls == [{"kind": "mp3", "bitrate_kbps": bitrate}]
    assert result.dtype == np.float32 and result.shape == waveform.shape


@pytest.mark.parametrize("condition", CONDITION_IDS[6:])
def test_low_pass_is_deterministic_and_preserves_contract(config, waveform, condition):
    first = apply_condition(waveform, condition_id=condition, clip_id="REAL/a.wav", config=config)
    second = apply_condition(waveform, condition_id=condition, clip_id="REAL/a.wav", config=config)
    assert first.tobytes() == second.tobytes()
    assert first.dtype == np.float32 and first.shape == waveform.shape
    assert np.isfinite(first).all()


def test_inputs_and_unknown_conditions_fail_closed(config, waveform):
    with pytest.raises(RobustnessError, match="float32"):
        apply_condition(waveform.astype(np.float64), condition_id=CONDITION_IDS[0], clip_id="x",
                        config=config)
    with pytest.raises(RobustnessError, match="unknown condition"):
        apply_condition(waveform, condition_id="clean", clip_id="x", config=config)
    with pytest.raises(RobustnessError, match="requires"):
        apply_condition(waveform, condition_id="mp3_32kbps", clip_id="x", config=config)
    with pytest.raises(RobustnessError, match="zero-power"):
        apply_condition(np.zeros(32_000, dtype=np.float32), condition_id=CONDITION_IDS[0],
                        clip_id="x", config=config)


def _clean_score_world(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_bytes(b"path,label\nREAL/a.wav,REAL\n")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(stage14_module, "F2_MANIFEST_SHA256", manifest_sha)
    registry_path = tmp_path / "registry.jsonl"
    registry = RunRegistry(registry_path)
    run_id = registry.claim("for", "test", MODEL_IDS, "a" * 40,
                            evaluation_purpose="stage11_test")
    registry.complete(run_id, "stage11/run_report.json")
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    return manifest, manifest_sha, registry_path, run_id, score_dir


def _write_clean(score_dir, run_id, manifest_sha, *, purpose="stage11_test",
                 model_id="baseline-v1", role="baseline"):
    kwargs = dict(
        evaluation_purpose=purpose, run_id=run_id, model_id=model_id, model_role=role,
        model_artifact_sha256="b" * 64, manifest_sha256=manifest_sha,
        clip_id=["REAL/a.wav"], label=["REAL"], group_id=["g1"], raw_score=[0.1],
    )
    if role == "baseline":
        kwargs["prediction"] = ["authentic"]
    elif role == "cnn_bundle_seed":
        kwargs.update(p=[0.2], prediction=["authentic"])
    write_clip_scores(score_dir, **kwargs)


def _write_stage11_report(tmp_path, score_dir, run_id, *, model_id="baseline-v1",
                          role="baseline"):
    artifact = "b" * 64
    roles = {
        "baseline-v1": "baseline",
        "cnn-v1-seed0": "cnn_bundle_seed",
        "cnn-v1-seed1": "cnn_other_seed",
        "cnn-v1-seed2": "cnn_other_seed",
    }
    report = {
        "schema_version": 1,
        "dataset": "for",
        "split": "test",
        "evaluation_purpose": "stage11_test",
        "run_id": run_id,
        "code_commit": "a" * 40,
        "model_set": {
            name: {"role": value, "artifact_sha256": artifact}
            for name, value in roles.items()
        },
        "clip_scores": {"folder": "clip_scores", "files": {
            model_id: {
                "csv_sha256": hashlib.sha256(
                    (score_dir / f"{model_id}.csv").read_bytes()).hexdigest(),
                "sidecar_sha256": hashlib.sha256(
                    (score_dir / f"{model_id}.json").read_bytes()).hexdigest(),
                "n_rows": 1,
            }
        }},
    }
    report["model_set"][model_id] = {"role": role, "artifact_sha256": artifact}
    path = tmp_path / f"{model_id}-run-report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_clean_score_reader_binds_f2_sidecar_and_completed_stage11_run(tmp_path, monkeypatch):
    manifest, manifest_sha, registry, run_id, scores = _clean_score_world(tmp_path, monkeypatch)
    _write_clean(scores, run_id, manifest_sha)
    report = _write_stage11_report(tmp_path, scores, run_id)
    loaded = read_stage11_clean_scores(
        scores, model_id="baseline-v1", registry_path=registry, manifest_path=manifest,
        run_report_path=report)
    assert loaded.metadata["run_id"] == run_id
    assert loaded.clip_id == ("REAL/a.wav",)


def test_clean_score_reader_refuses_uncompleted_or_wrong_run(tmp_path, monkeypatch):
    manifest, manifest_sha, registry_path, _, scores = _clean_score_world(tmp_path, monkeypatch)
    _write_clean(scores, "f" * 32, manifest_sha)
    with pytest.raises(RobustnessError, match="0 matching completed"):
        read_stage11_clean_scores(
            scores, model_id="baseline-v1", registry_path=registry_path, manifest_path=manifest,
            run_report_path=tmp_path / "not-read.json")


def test_clean_score_reader_refuses_a_partial_stage11_model_set(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_bytes(b"path,label\nREAL/a.wav,REAL\n")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(stage14_module, "F2_MANIFEST_SHA256", manifest_sha)
    registry_path = tmp_path / "registry.jsonl"
    registry = RunRegistry(registry_path)
    run_id = registry.claim(
        "for", "test", ("baseline-v1",), "a" * 40, evaluation_purpose="stage11_test")
    registry.complete(run_id, "stage11/run_report.json")
    scores = tmp_path / "scores"
    scores.mkdir()
    _write_clean(scores, run_id, manifest_sha)
    with pytest.raises(RobustnessError, match="0 matching completed"):
        read_stage11_clean_scores(
            scores, model_id="baseline-v1", registry_path=registry_path, manifest_path=manifest,
            run_report_path=tmp_path / "not-read.json")


def test_clean_score_reader_refuses_wrong_manifest_or_non_primary_role(tmp_path, monkeypatch):
    manifest, manifest_sha, registry, run_id, scores = _clean_score_world(tmp_path, monkeypatch)
    _write_clean(scores, run_id, "c" * 64)
    with pytest.raises(RobustnessError, match="differs from F2"):
        read_stage11_clean_scores(
            scores, model_id="baseline-v1", registry_path=registry, manifest_path=manifest,
            run_report_path=tmp_path / "not-read.json")

    other = tmp_path / "other"
    other.mkdir()
    _write_clean(other, run_id, manifest_sha, model_id="cnn-v1-seed0", role="cnn_other_seed")
    with pytest.raises(RobustnessError, match="only baseline or frozen CNN bundle"):
        read_stage11_clean_scores(
            other, model_id="cnn-v1-seed0", registry_path=registry, manifest_path=manifest,
            run_report_path=tmp_path / "not-read.json")


def test_clean_score_reader_refuses_files_not_bound_by_stage11_report(tmp_path, monkeypatch):
    manifest, manifest_sha, registry, run_id, scores = _clean_score_world(tmp_path, monkeypatch)
    _write_clean(scores, run_id, manifest_sha)
    report = _write_stage11_report(tmp_path, scores, run_id)
    data = json.loads(report.read_text(encoding="utf-8"))
    data["clip_scores"]["files"]["baseline-v1"]["csv_sha256"] = "f" * 64
    report.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RobustnessError, match="hashes/count"):
        read_stage11_clean_scores(
            scores, model_id="baseline-v1", registry_path=registry, manifest_path=manifest,
            run_report_path=report)


def _scores(*, role, purpose, raw, prediction=None, clip_ids=None):
    clip_ids = tuple(f"clip/{i}.wav" for i in range(8)) if clip_ids is None else tuple(clip_ids)
    return ClipScores(
        metadata={
            "evaluation_purpose": purpose,
            "run_id": ("a" if purpose == "stage11_test" else "b") * 32,
            "model_id": "baseline-v1" if role == "baseline" else "cnn-v1-seed1",
            "model_role": role,
            "model_artifact_sha256": "c" * 64,
            "manifest_sha256": "d" * 64,
            "n_rows": 8,
            "csv_sha256": "e" * 64,
            "applicable_columns": [],
        },
        clip_id=clip_ids,
        label=("REAL", "FAKE", "REAL", "FAKE", "REAL", "FAKE", "REAL", "FAKE"),
        group_id=tuple(f"g{i}" for i in range(8)),
        raw_score=np.asarray(raw, dtype=np.float64),
        p=None,
        prediction=None if prediction is None else tuple(prediction),
    )


def test_paired_baseline_comparison_reports_primary_endpoint(config):
    clean = _scores(
        role="baseline", purpose="stage11_test",
        raw=[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
        prediction=["authentic", "synthetic"] * 4,
    )
    degraded = _scores(
        role="baseline", purpose="stage14_robustness",
        raw=[0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5],
        prediction=["synthetic", "authentic"] * 4,
    )
    result = compare_condition(
        clean, degraded, condition_id="low_pass_3400hz", config=config)
    assert result["n_rows"] == 8
    assert result["eer_increase"]["estimate"] == pytest.approx(
        result["degraded_eer"] - result["clean_eer"])
    assert result["eer_increase"]["n_resamples"] == 2000
    assert isinstance(result["primary_supported"], bool)


def test_paired_bundle_comparison_reports_h6_real_false_synthetic(config):
    clean_predictions = ["authentic", "synthetic"] * 4
    degraded_predictions = ["synthetic", "synthetic"] * 4
    clean = _scores(
        role="cnn_bundle_seed", purpose="stage11_test",
        raw=[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6], prediction=clean_predictions)
    degraded = _scores(
        role="cnn_bundle_seed", purpose="stage14_robustness",
        raw=[0.8, 0.9, 0.7, 0.8, 0.6, 0.7, 0.5, 0.6], prediction=degraded_predictions)
    result = compare_condition(
        clean, degraded, condition_id="low_pass_3400hz", config=config)
    assert result["clean_real_false_synthetic_rate"] == 0.0
    assert result["degraded_real_false_synthetic_rate"] == 1.0
    assert result["real_false_synthetic_rate_increase"]["estimate"] == 1.0
    assert result["h6_supported"] is True


def test_paired_comparison_refuses_row_or_identity_drift(config):
    clean = _scores(
        role="baseline", purpose="stage11_test",
        raw=[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    degraded = _scores(
        role="baseline", purpose="stage14_robustness",
        raw=[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
        clip_ids=["wrong.wav", *clean.clip_id[1:]])
    with pytest.raises(RobustnessError, match="clip_id rows differ"):
        compare_condition(clean, degraded, condition_id=CONDITION_IDS[0], config=config)

    clean.metadata["model_role"] = "cnn_other_seed"
    degraded = _scores(
        role="baseline", purpose="stage14_robustness",
        raw=[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    degraded.metadata["model_role"] = "cnn_other_seed"
    with pytest.raises(RobustnessError, match="does not compare model role"):
        compare_condition(clean, degraded, condition_id=CONDITION_IDS[0], config=config)
