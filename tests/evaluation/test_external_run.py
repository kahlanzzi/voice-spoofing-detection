"""Synthetic-only coverage for Stage 13's guarded preparation primitives."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from src.evaluation import external_run
from src.preprocessing import load_profile, preprocess_waveform
from tests._torch import require_torch


@pytest.fixture(scope="module")
def profile():
    return load_profile(
        Path(__file__).resolve().parents[2] / "configs" / "preprocessing.yaml"
    )


def _fake_artifacts(profile=None, *, v2_model=None, v2_sha=None):
    policy = SimpleNamespace(
        threshold=0.5,
        half_width=0.1,
        decide=lambda values: tuple(
            SimpleNamespace(
                p=0.75 if value >= 0 else 0.25,
                prediction="synthetic" if value >= 0 else "authentic",
            )
            for value in values
        ),
    )
    bundle = SimpleNamespace(
        profile=profile or SimpleNamespace(fingerprint=lambda: "f" * 64),
        policy=policy,
        manifest=SimpleNamespace(bundle_version="bundle-v1"),
    )
    return external_run.FrozenArtifacts(
        SimpleNamespace(bundle=bundle),
        SimpleNamespace(threshold=0.0),
        "a" * 64,
        "b" * 64,
        v2_model,
        v2_sha,
    )


def _stub_prepare_gates(monkeypatch, calls):
    require_torch()
    from src.evaluation import stage11_run

    monkeypatch.setattr(
        stage11_run,
        "environment_record",
        lambda: ({"synthetic": True}, "synthetic==1\n"),
    )
    monkeypatch.setattr(
        external_run,
        "verify_d58_rehearsal_report",
        lambda *args, **kwargs: calls.append("d58") or "e" * 64,
    )

    def fake_scorers(artifacts):
        calls.append("scorers")
        result = {"bundle-v1": lambda _: 0.1, "baseline-v1": lambda _: 0.2}
        if artifacts.v2_model is not None:
            result["bundle-v2"] = lambda _: 0.3
        return result

    monkeypatch.setattr(external_run, "build_frozen_scorers", fake_scorers)


def _synthetic_d58_report(baseline_sha, profile_sha, environment):
    def passed(actual=None, expected="recorded"):
        return {"expected": expected, "actual": actual, "passed": True}

    def mfcc_checks(split):
        reference = external_run.D58_MFCC_REFERENCE_SHA256[split]
        return {
            f"{split}.mfcc.reference_sha256": passed(reference, reference),
            f"{split}.mfcc.reference_files": passed(
                {
                    f"{split}.npz": reference,
                    f"{split}.json": hashlib.sha256(
                        f"{split}-rows".encode()
                    ).hexdigest(),
                }
            ),
            f"{split}.mfcc.reference_rows": passed(
                {"n_rows": 4, "rows_sha256": "1" * 64}
            ),
            f"{split}.mfcc.within_tolerance": passed(
                {
                    "allclose": True,
                    "rtol": external_run.D58_MFCC_RTOL,
                    "atol": external_run.D58_MFCC_ATOL,
                }
            ),
        }

    checks = {
        "baseline.artifact_sha256": passed(baseline_sha, baseline_sha),
        "profile_fingerprint": passed(profile_sha, profile_sha),
        "reproduction.mfcc.val-A.reference_sha256": passed(
            external_run.D58_MFCC_REFERENCE_SHA256["val-A"],
            external_run.D58_MFCC_REFERENCE_SHA256["val-A"],
        ),
        "reproduction.mfcc.val-A.reference_files": passed(
            {
                "val-A.npz": external_run.D58_MFCC_REFERENCE_SHA256["val-A"],
                "val-A.json": hashlib.sha256(b"val-A-rows").hexdigest(),
            }
        ),
        "reproduction.mfcc.val-A.reference_rows": passed(
            {"n_rows": 4, "rows_sha256": "2" * 64}
        ),
        "reproduction.mfcc.val-A.within_tolerance": passed(
            {
                "allclose": True,
                "rtol": external_run.D58_MFCC_RTOL,
                "atol": external_run.D58_MFCC_ATOL,
            }
        ),
    }
    known_answers = {}
    for split in external_run.D58_MFCC_REFERENCE_SHA256:
        known_answers.update(mfcc_checks(split))
        known_answers[f"baseline.{split}.predictions_equal_reference"] = passed(0, 0)
        known_answers[f"baseline.{split}.max_abs_score_diff"] = passed(1e-7)
        known_answers[f"baseline.{split}.near_threshold_clips"] = passed(0)
    return {
        "schema_version": 1,
        "passed": True,
        "model_set": {
            "baseline-v1": {"artifact_sha256": baseline_sha, "role": "baseline"}
        },
        "environment": environment,
        "scoring": {"device": "cpu"},
        "pre_claim_checks": checks,
        "known_answers": known_answers,
    }


def _tone(samples: int, amplitude: float = 0.2) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32) / 16000.0
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _window(silence_each_side: int) -> np.ndarray:
    voiced = 32000 - 2 * silence_each_side
    return np.concatenate(
        (
            np.zeros(silence_each_side, dtype=np.float32),
            _tone(voiced),
            np.zeros(silence_each_side, dtype=np.float32),
        )
    )


def test_preprocesses_each_complete_window_before_scoring_and_drops_tail(
    monkeypatch, profile
):
    windows = (_window(4000), _window(8000), _window(12000))
    tail = np.zeros(4000, dtype=np.float32)
    recording = np.concatenate((*windows, tail))
    actual_inputs = []
    scorer_inputs = []
    real_preprocess = preprocess_waveform

    def wrapped(raw, sample_rate, supplied_profile, **kwargs):
        actual_inputs.append(np.array(raw, copy=True))
        return real_preprocess(raw, sample_rate, supplied_profile, **kwargs)

    monkeypatch.setattr(external_run, "preprocess_waveform", wrapped)

    def fake_scorer(prepared):
        scorer_inputs.append(np.array(prepared, copy=True))
        return float(np.mean(prepared))

    result = external_run.score_windows(recording, 16000, profile, fake_scorer)
    assert result.n_complete_windows == 3
    assert result.n_dropped_samples == len(tail)
    assert result.n_filter_dropped == 1
    assert tuple(row.window_index for row in result.filtered_scores) == (0, 1)
    assert len(actual_inputs) == len(scorer_inputs) == 3
    for actual, expected in zip(actual_inputs, windows):
        np.testing.assert_array_equal(actual, expected)
    for raw, actual in zip(windows, scorer_inputs):
        np.testing.assert_array_equal(
            actual, real_preprocess(raw, 16000, profile).waveform
        )
    assert tuple(row.padding_samples for row in result.scores) == (7680, 16000, 23680)


def test_joint_scorers_share_each_preprocessed_window(monkeypatch, profile):
    windows = (_window(4000), _window(8000))
    recording = np.concatenate(windows)
    calls = {"preprocess": [], "v1": [], "baseline": []}
    real_preprocess = preprocess_waveform

    def wrapped(raw, sample_rate, supplied_profile, **kwargs):
        calls["preprocess"].append(np.array(raw, copy=True))
        return real_preprocess(raw, sample_rate, supplied_profile, **kwargs)

    monkeypatch.setattr(external_run, "preprocess_waveform", wrapped)

    def scorer(name, offset):
        def score(prepared):
            calls[name].append(np.array(prepared, copy=True))
            return float(np.mean(prepared) + offset)

        return score

    result = external_run.score_window_models(
        recording,
        16000,
        profile,
        {"v1": scorer("v1", 0), "baseline": scorer("baseline", 1)},
    )
    assert len(calls["preprocess"]) == len(calls["v1"]) == len(calls["baseline"]) == 2
    for first, second in zip(calls["v1"], calls["baseline"]):
        np.testing.assert_array_equal(first, second)
    assert result["v1"].n_complete_windows == result["baseline"].n_complete_windows == 2
    np.testing.assert_allclose(
        [row.raw_score for row in result["baseline"].scores],
        [row.raw_score + 1 for row in result["v1"].scores],
    )


def test_f4_preflight_checks_both_anchors_before_checkpoint_model_load(tmp_path):
    record_path = tmp_path / "f4.json"
    record_path.write_text(
        json.dumps(
            {
                "checks": [{"ok": True}],
                "manifest_sha256": "a" * 64,
                "bundle_version": "bundle-v1",
                "baseline": {"artifact_sha256": "b" * 64, "version": "baseline-v1"},
            }
        ),
        encoding="utf-8",
    )
    bundle = SimpleNamespace(
        manifest_sha256="a" * 64, manifest=SimpleNamespace(bundle_version="bundle-v1")
    )
    calls = []
    loaded_bundle_model = SimpleNamespace(bundle=bundle)
    baseline_model = SimpleNamespace(version="baseline-v1")

    got = external_run.verify_f4_artifacts(
        tmp_path / "bundle",
        tmp_path / "baseline",
        record_path,
        bundle_verifier=lambda path: calls.append("verify-bundle") or bundle,
        baseline_hash=lambda path: "b" * 64,
        baseline_loader=lambda path: calls.append("load-baseline") or baseline_model,
        bundle_loader=lambda path: calls.append("load-cnn") or loaded_bundle_model,
    )
    assert calls == ["verify-bundle", "load-baseline", "load-cnn"]
    assert got.bundle_manifest_sha256 == "a" * 64
    assert got.baseline_artifact_sha256 == "b" * 64


def test_f4_preflight_refuses_mismatch_before_loading_models(tmp_path):
    record_path = tmp_path / "f4.json"
    record_path.write_text(
        json.dumps(
            {
                "checks": [{"ok": True}],
                "manifest_sha256": "a" * 64,
                "bundle_version": "bundle-v1",
                "baseline": {"artifact_sha256": "b" * 64, "version": "baseline-v1"},
            }
        ),
        encoding="utf-8",
    )
    bundle = SimpleNamespace(
        manifest_sha256="wrong", manifest=SimpleNamespace(bundle_version="bundle-v1")
    )
    model_loader_called = False

    def forbidden_loader(_):
        nonlocal model_loader_called
        model_loader_called = True
        raise AssertionError("must not load a checkpoint after failed preflight")

    with pytest.raises(ValueError, match="manifest does not match"):
        external_run.verify_f4_artifacts(
            tmp_path / "bundle",
            tmp_path / "baseline",
            record_path,
            bundle_verifier=lambda path: bundle,
            baseline_hash=lambda path: "b" * 64,
            baseline_loader=lambda path: SimpleNamespace(version="baseline-v1"),
            bundle_loader=forbidden_loader,
        )
    assert not model_loader_called


def test_claim_is_atomic_dataset_wide_across_models_and_failures_persist(tmp_path):
    barrier = threading.Barrier(2)
    outcomes = []

    def attempt(run_id, models):
        barrier.wait()
        try:
            outcomes.append(
                external_run.claim_dataset_once(
                    tmp_path,
                    dataset="DEEP-VOICE",
                    run_id=run_id,
                    model_artifacts=models,
                )
            )
        except RuntimeError as exc:
            outcomes.append(str(exc))

    threads = [
        threading.Thread(target=attempt, args=("run-a", {"v1": "a" * 64})),
        threading.Thread(
            target=attempt, args=("run-b", {"baseline": "b" * 64, "v2": "c" * 64})
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    claims = [outcome for outcome in outcomes if isinstance(outcome, tuple)]
    refusals = [outcome for outcome in outcomes if isinstance(outcome, str)]
    assert len(claims) == len(refusals) == 1
    claim_path, _ = claims[0]
    content = json.loads(claim_path.read_text(encoding="utf-8"))
    final_hash = external_run.finalize_claim(
        claim_path,
        run_id=content["run_id"],
        status="failed",
        detail="synthetic test failure",
    )
    assert len(final_hash) == 64
    assert json.loads(claim_path.read_text(encoding="utf-8"))["status"] == "failed"
    with pytest.raises(RuntimeError, match="already exists"):
        external_run.claim_dataset_once(
            tmp_path,
            dataset="DEEP-VOICE",
            run_id="new-run",
            model_artifacts={"v1": "a" * 64},
        )


def test_claim_rejects_dataset_alias_that_could_create_a_second_key(tmp_path):
    with pytest.raises(ValueError, match="exactly 'DEEP-VOICE'"):
        external_run.claim_dataset_once(
            tmp_path,
            dataset="deep-voice",
            run_id="alias",
            model_artifacts={"v1": "a" * 64},
        )


def test_manifest_hash_is_checked_without_audio_access(tmp_path):
    manifest = tmp_path / "metadata.csv"
    manifest.write_text("path,label\nplaceholder.wav,FAKE\n", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert external_run.verify_pinned_manifest(manifest, digest) == digest
    with pytest.raises(ValueError, match="mismatch"):
        external_run.verify_pinned_manifest(manifest, "0" * 64)


def _synthetic_external_manifest() -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        (
            "path",
            "label",
            "source_speaker",
            "target_speaker",
            "sample_rate",
            "channels",
            "duration_s",
            "hash",
        )
    )
    speakers = [f"speaker-{i}" for i in range(8)]
    for index, speaker in enumerate(speakers):
        writer.writerow(
            (f"AUDIO/REAL/{speaker}.wav", "REAL", speaker, "", 44100, 2, 5.0, "a" * 64)
        )
        for target in speakers:
            if target != speaker:
                writer.writerow(
                    (
                        f"AUDIO/FAKE/{speaker}-to-{target}.wav",
                        "FAKE",
                        speaker,
                        target,
                        44100,
                        2,
                        5.0,
                        "b" * 64,
                    )
                )
    return buffer.getvalue().encode("utf-8")


def test_external_manifest_is_hash_pinned_and_metadata_only(tmp_path):
    content = _synthetic_external_manifest()
    path = tmp_path / "manifest.csv"
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    rows, actual = external_run.load_external_manifest(path, digest)
    assert len(rows) == 64
    assert actual == digest
    assert sum(row.label == "FAKE" for row in rows) == 56
    assert len({row.source_speaker for row in rows}) == 8
    assert rows[0].sample_rate == 44100
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        external_run.load_external_manifest(path, "0" * 64)


def test_external_manifest_rejects_unapproved_counts(tmp_path):
    content = _synthetic_external_manifest()
    path = tmp_path / "manifest.csv"
    path.write_bytes(content[:-20])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="manifest row|64 unique"):
        external_run.load_external_manifest(path, digest)


def test_prepare_guarded_run_verifies_prerequisites_then_claims_before_return(
    monkeypatch, tmp_path
):
    calls = []
    artifacts = _fake_artifacts()
    _stub_prepare_gates(monkeypatch, calls)
    manifest_files = (
        external_run.ManifestFile("AUDIO/REAL/a.wav", "REAL", "a", None, 2.0),
    )
    monkeypatch.setattr(
        external_run,
        "load_external_manifest",
        lambda path, digest: calls.append("manifest") or (manifest_files, digest),
    )
    monkeypatch.setattr(
        external_run,
        "verify_f4_artifacts",
        lambda *args, **kwargs: calls.append("frozen-artifacts") or artifacts,
    )

    def claim(*args, **kwargs):
        calls.append("durable-claim")
        path = tmp_path / "claim.json"
        path.write_text("{}", encoding="utf-8")
        return path, "c" * 64

    monkeypatch.setattr(external_run, "claim_dataset_once", claim)
    monkeypatch.setattr(
        external_run,
        "_tunis_now",
        lambda: datetime(2026, 12, 5, tzinfo=ZoneInfo("Africa/Tunis")),
    )
    monkeypatch.setattr(
        external_run,
        "verify_d58_rehearsal_report",
        lambda *args, **kwargs: calls.append("d58") or "e" * 64,
    )
    prepared = external_run.prepare_guarded_run(
        {
            "dataset": "DEEP-VOICE",
            "registry_dir": tmp_path,
            "bundle_dir": tmp_path,
            "baseline_dir": tmp_path,
            "f4_record": tmp_path,
            "manifest_path": tmp_path,
            "manifest_sha256": "d" * 64,
            "d58_rehearsal_report": tmp_path / "d58.json",
            "d58_rehearsal_report_sha256": "e" * 64,
        },
        run_id="1" * 32,
    )
    assert calls == ["manifest", "frozen-artifacts", "d58", "scorers", "durable-claim"]
    assert prepared.artifacts is artifacts
    assert prepared.manifest_files == manifest_files
    assert prepared.claim_sha256 == "c" * 64
    assert prepared.model_set == ("bundle-v1", "baseline-v1")


def test_prepare_guarded_run_refuses_v1_only_before_d48_cutoff(monkeypatch, tmp_path):
    calls = []
    manifest_files = (
        external_run.ManifestFile("AUDIO/REAL/a.wav", "REAL", "a", None, 2.0),
    )
    artifacts = _fake_artifacts()
    monkeypatch.setattr(
        external_run,
        "load_external_manifest",
        lambda path, digest: calls.append("manifest") or (manifest_files, digest),
    )
    monkeypatch.setattr(
        external_run,
        "verify_f4_artifacts",
        lambda *args, **kwargs: calls.append("f4") or artifacts,
    )
    monkeypatch.setattr(
        external_run,
        "claim_dataset_once",
        lambda *args, **kwargs: (
            calls.append("claim") or (tmp_path / "claim.json", "c" * 64)
        ),
    )
    monkeypatch.setattr(
        external_run,
        "_tunis_now",
        lambda: datetime(2026, 11, 15, tzinfo=ZoneInfo("Africa/Tunis")),
    )

    with pytest.raises(RuntimeError, match="D48 requires verified F5"):
        external_run.prepare_guarded_run(
            {
                "dataset": "DEEP-VOICE",
                "registry_dir": tmp_path,
                "bundle_dir": tmp_path,
                "baseline_dir": tmp_path,
                "f4_record": tmp_path,
                "manifest_path": tmp_path,
                "manifest_sha256": "d" * 64,
                "d58_rehearsal_report": tmp_path / "d58.json",
                "d58_rehearsal_report_sha256": "e" * 64,
            },
            run_id="1" * 32,
        )
    assert calls == ["manifest", "f4"]


def test_prepare_guarded_run_includes_verified_f5_in_one_claim(monkeypatch, tmp_path):
    calls = []
    f5_sha = "e" * 64
    manifest_files = (
        external_run.ManifestFile("AUDIO/REAL/a.wav", "REAL", "a", None, 2.0),
    )
    artifacts = _fake_artifacts()
    _stub_prepare_gates(monkeypatch, calls)
    bundle = SimpleNamespace(
        manifest_sha256=f5_sha,
        manifest=SimpleNamespace(bundle_version="bundle-v2"),
    )
    v2_model = SimpleNamespace(bundle=bundle)
    f5_path = tmp_path / "f5.json"
    f5_path.write_text(
        json.dumps(
            {
                "checks": [{"ok": True}],
                "manifest_sha256": f5_sha,
                "bundle_version": "bundle-v2",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        external_run,
        "load_external_manifest",
        lambda path, digest: calls.append("manifest") or (manifest_files, digest),
    )
    monkeypatch.setattr(
        external_run,
        "verify_f4_artifacts",
        lambda *args, **kwargs: calls.append("f4") or artifacts,
    )

    def claim(*args, **kwargs):
        calls.append("claim")
        assert kwargs["model_artifacts"]["bundle-v2"] == f5_sha
        claim_path = tmp_path / "claim.json"
        claim_path.write_text("{}", encoding="utf-8")
        return claim_path, "c" * 64

    monkeypatch.setattr(external_run, "claim_dataset_once", claim)
    monkeypatch.setattr(
        external_run,
        "_tunis_now",
        lambda: datetime(2026, 11, 15, tzinfo=ZoneInfo("Africa/Tunis")),
    )
    prepared = external_run.prepare_guarded_run(
        {
            "dataset": "DEEP-VOICE",
            "registry_dir": tmp_path,
            "bundle_dir": tmp_path,
            "baseline_dir": tmp_path,
            "f4_record": tmp_path,
            "v2_bundle_dir": tmp_path / "v2",
            "f5_record": f5_path,
            "manifest_path": tmp_path,
            "manifest_sha256": "d" * 64,
            "d58_rehearsal_report": tmp_path / "d58.json",
            "d58_rehearsal_report_sha256": "e" * 64,
        },
        run_id="2" * 32,
        f5_verifier=lambda path: calls.append("verify-f5") or bundle,
        f5_loader=lambda path: calls.append("load-f5") or v2_model,
    )
    assert calls == [
        "manifest",
        "f4",
        "verify-f5",
        "load-f5",
        "d58",
        "scorers",
        "claim",
    ]
    assert prepared.artifacts.v2_model is v2_model
    assert prepared.artifacts.v2_manifest_sha256 == f5_sha
    assert prepared.model_set == ("bundle-v1", "bundle-v2", "baseline-v1")


def test_d58_rehearsal_is_hash_pinned_and_requires_the_same_runtime(tmp_path):
    baseline_sha, profile_sha = "b" * 64, "f" * 64
    environment = {
        "pip_freeze_sha256": "a" * 64,
        "packages": {"torch": "2.11.0+cu130"},
        "cpu": {"model": "Synthetic CPU"},
        "gpu": "Tesla T4 synthetic driver",
        "platform": "Synthetic Linux",
        "python": "3.13.16",
        "failures": {},
    }
    report_path = tmp_path / "synthetic-d58.json"
    report_path.write_text(
        json.dumps(_synthetic_d58_report(baseline_sha, profile_sha, environment)),
        encoding="utf-8",
    )
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()

    assert (
        external_run.verify_d58_rehearsal_report(
            report_path,
            report_sha,
            baseline_artifact_sha256=baseline_sha,
            profile_fingerprint=profile_sha,
            current_environment=environment,
        )
        == report_sha
    )
    with pytest.raises(ValueError, match="exact current runtime"):
        external_run.verify_d58_rehearsal_report(
            report_path,
            report_sha,
            baseline_artifact_sha256=baseline_sha,
            profile_fingerprint=profile_sha,
            current_environment={**environment, "gpu": "Different GPU"},
        )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        external_run.verify_d58_rehearsal_report(
            report_path,
            "0" * 64,
            baseline_artifact_sha256=baseline_sha,
            profile_fingerprint=profile_sha,
            current_environment=environment,
        )


def test_h8_pairs_two_cnn_models_and_preserves_d36_default_role(tmp_path):
    manifest = tuple(
        external_run.ManifestFile(
            f"AUDIO/{'FAKE' if i % 2 else 'REAL'}/clip-{i}.wav",
            "FAKE" if i % 2 else "REAL",
            f"speaker-{i}",
            None,
            2.0,
        )
        for i in range(8)
    )
    run_id = "1" * 32
    v1_rows, v2_rows = [], []
    for index, item in enumerate(manifest):
        fake = item.label == "FAKE"
        v1_score = 0.9 if fake else 0.1
        v2_score = 0.1 if fake else 0.9
        common = {
            "file": item.file,
            "window_index": 0,
            "source_speaker": item.source_speaker,
            "target_speaker": None,
            "label": item.label,
            "kept_by_filter": index != 7,
            "run_id": run_id,
        }
        v1_rows.append(
            external_run.ExternalScoreRow(
                **common,
                raw_score=v1_score,
                calibrated_p=v1_score,
                prediction="synthetic" if fake else "authentic",
                bundle_version="bundle-v1",
            )
        )
        v2_rows.append(
            external_run.ExternalScoreRow(
                **common,
                raw_score=v2_score,
                calibrated_p=v2_score,
                prediction="synthetic" if not fake else "authentic",
                bundle_version="bundle-v2",
            )
        )
    manifest_sha = "d" * 64
    v1_paths = external_run.write_external_score_file(
        tmp_path / "v1",
        model_id="bundle-v1",
        model_role="cnn_bundle",
        model_artifact_sha256="a" * 64,
        manifest_sha256=manifest_sha,
        rows=v1_rows,
    )
    v2_paths = external_run.write_external_score_file(
        tmp_path / "v2",
        model_id="bundle-v2",
        model_role="cnn_bundle",
        model_artifact_sha256="e" * 64,
        manifest_sha256=manifest_sha,
        rows=v2_rows,
    )
    v1_file = external_run.read_external_score_file(*v1_paths)
    v2_file = external_run.read_external_score_file(*v2_paths)

    with pytest.raises(ValueError, match="expected a baseline"):
        external_run.paired_eer_comparison(
            v1_file, v2_file, manifest, view="filtered", n_resamples=40
        )
    primary = external_run.paired_eer_comparison(
        v1_file,
        v2_file,
        manifest,
        view="filtered",
        n_resamples=40,
        second_model_role="cnn_bundle",
    )
    sensitivity = external_run.paired_eer_comparison(
        v1_file,
        v2_file,
        manifest,
        view="unfiltered",
        n_resamples=40,
        second_model_role="cnn_bundle",
    )
    assert primary.n_windows == 7
    assert sensitivity.n_windows == 8
    assert primary.difference == pytest.approx(primary.cnn_eer - primary.baseline_eer)
    assert primary.difference < 0


def _synthetic_prepared_run(tmp_path, profile, manifest_files, *, run_id="2" * 32):
    artifacts = _fake_artifacts(profile)
    model_artifacts = {
        "baseline-v1": artifacts.baseline_artifact_sha256,
        "bundle-v1": artifacts.bundle_manifest_sha256,
        "deepvoice-manifest": "c" * 64,
        "d58-rehearsal-report": "d" * 64,
    }
    claim_path, claim_sha = external_run.claim_dataset_once(
        tmp_path / "registry",
        dataset="DEEP-VOICE",
        run_id=run_id,
        model_artifacts=model_artifacts,
    )
    prepared = external_run.PreparedRun(
        artifacts,
        tuple(manifest_files),
        "c" * 64,
        claim_path,
        claim_sha,
        "d" * 64,
        {"bundle-v1": lambda _: 0.1, "baseline-v1": lambda _: 0.2},
        run_id,
        ("bundle-v1", "baseline-v1"),
    )
    return prepared


def test_guarded_scoring_consumes_synthetic_files_once_and_writes_report(
    tmp_path, profile
):
    manifest, expected_bytes = [], {}
    for speaker_index in range(8):
        speaker = f"speaker-{speaker_index}"
        for label in ("REAL", "FAKE"):
            filename = f"AUDIO/{label}/{speaker}.wav"
            data = f"synthetic-only-{label}-{speaker}".encode()
            expected_bytes[filename] = data
            manifest.append(
                external_run.ManifestFile(
                    filename,
                    label,
                    speaker,
                    None if label == "REAL" else f"target-{speaker_index}",
                    2.0,
                    16000,
                    1,
                    hashlib.sha256(data).hexdigest(),
                )
            )
    prepared = _synthetic_prepared_run(tmp_path, profile, manifest)
    audio_root = tmp_path / "synthetic-audio"
    for name, data in expected_bytes.items():
        path = audio_root.joinpath(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    decoded = []

    def synthetic_decoder(data, name):
        assert expected_bytes[name] == data
        decoded.append(name)
        return _window(4000), 16000

    result = external_run.score_guarded_dataset(
        prepared,
        audio_root,
        output_dir=tmp_path / "scores",
        decoder=synthetic_decoder,
        n_resamples=40,
        seed=12,
    )
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    claim = json.loads(result.claim_path.read_text(encoding="utf-8"))
    assert result.files_scored == len(manifest) == len(decoded)
    assert result.complete_windows == len(manifest)
    assert set(result.score_files) == {"bundle-v1", "baseline-v1"}
    assert report["hypotheses"]["H8"]["status"] == "not_tested_cutoff_excluded_v2"
    assert claim["status"] == "succeeded"


def test_guarded_scoring_finalizes_missing_synthetic_clip_as_incomplete(
    tmp_path, profile
):
    item = external_run.ManifestFile(
        "AUDIO/REAL/missing.wav",
        "REAL",
        "speaker-0",
        None,
        2.0,
        16000,
        1,
        hashlib.sha256(b"never-created").hexdigest(),
    )
    prepared = _synthetic_prepared_run(tmp_path, profile, [item])
    audio_root = tmp_path / "empty-synthetic-audio"
    audio_root.mkdir()

    with pytest.raises(FileNotFoundError):
        external_run.score_guarded_dataset(
            prepared,
            audio_root,
            output_dir=tmp_path / "failed-scores",
            decoder=lambda *_: pytest.fail("missing file must fail before decode"),
        )
    claim = json.loads(prepared.claim_path.read_text(encoding="utf-8"))
    assert claim["status"] == "incomplete"
    assert "FileNotFoundError" in claim["detail"]


def test_config_template_is_strict_and_does_not_accept_placeholders():
    path = Path(__file__).resolve().parents[2] / "configs" / "external_run.yaml"
    with pytest.raises(ValueError, match="concrete"):
        external_run.load_external_config(path)


def test_external_score_files_share_schema_and_refuse_overwrite(tmp_path):
    run_id = "1" * 32
    common = {
        "file": "AUDIO/FAKE/source-to-target.wav",
        "window_index": 1,
        "source_speaker": "source",
        "target_speaker": "target",
        "label": "FAKE",
        "raw_score": 0.75,
        "kept_by_filter": True,
        "run_id": run_id,
    }
    cnn_row = external_run.ExternalScoreRow(
        **common, calibrated_p=0.8, prediction="synthetic", bundle_version="bundle-v1"
    )
    baseline_row = external_run.ExternalScoreRow(
        **common, calibrated_p=None, prediction="synthetic", bundle_version=None
    )
    cnn_csv, cnn_json = external_run.write_external_score_file(
        tmp_path / "cnn",
        model_id="bundle-v1",
        model_role="cnn_bundle",
        model_artifact_sha256="a" * 64,
        manifest_sha256="c" * 64,
        rows=[cnn_row],
    )
    baseline_csv, baseline_json = external_run.write_external_score_file(
        tmp_path / "baseline",
        model_id="baseline-v1",
        model_role="baseline",
        model_artifact_sha256="b" * 64,
        manifest_sha256="c" * 64,
        rows=[baseline_row],
    )
    assert (
        cnn_csv.read_bytes().splitlines()[0]
        == baseline_csv.read_bytes().splitlines()[0]
    )
    cnn_sidecar, baseline_sidecar = (
        json.loads(cnn_json.read_text()),
        json.loads(baseline_json.read_text()),
    )
    assert (
        cnn_sidecar["columns"]
        == baseline_sidecar["columns"]
        == list(external_run.SCORE_COLUMNS)
    )
    assert "0.8" in cnn_csv.read_text(encoding="utf-8")
    assert ",,synthetic," in baseline_csv.read_text(encoding="utf-8")
    loaded_meta, loaded_rows = external_run.read_external_score_file(cnn_csv)
    assert loaded_meta["model_id"] == "bundle-v1"
    assert loaded_rows == (cnn_row,)
    baseline_meta, baseline_rows = external_run.read_external_score_file(baseline_csv)
    assert baseline_meta["model_role"] == "baseline"
    assert baseline_rows == (baseline_row,)
    with pytest.raises(FileExistsError):
        external_run.write_external_score_file(
            tmp_path / "cnn",
            model_id="bundle-v1",
            model_role="cnn_bundle",
            model_artifact_sha256="a" * 64,
            manifest_sha256="c" * 64,
            rows=[cnn_row],
        )


def test_external_score_reader_detects_csv_tampering(tmp_path):
    row = external_run.ExternalScoreRow(
        file="AUDIO/REAL/a.wav",
        window_index=0,
        source_speaker="a",
        target_speaker=None,
        label="REAL",
        raw_score=0.1,
        calibrated_p=0.1,
        prediction="authentic",
        kept_by_filter=True,
        bundle_version="bundle-v1",
        run_id="1" * 32,
    )
    csv_path, _ = external_run.write_external_score_file(
        tmp_path,
        model_id="v1",
        model_role="cnn_bundle",
        model_artifact_sha256="a" * 64,
        manifest_sha256="c" * 64,
        rows=[row],
    )
    csv_path.write_bytes(csv_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        external_run.read_external_score_file(csv_path)


def test_paired_eer_comparison_uses_shared_speaker_clustered_resamples(tmp_path):
    manifest = tuple(
        external_run.ManifestFile(
            f"AUDIO/{'FAKE' if i % 2 else 'REAL'}/clip-{i}.wav",
            "FAKE" if i % 2 else "REAL",
            f"speaker-{i}",
            None,
            2.0,
        )
        for i in range(8)
    )
    cnn_scores = (0.1, 0.9, 0.3, 0.7, 0.8, 0.2, 0.4, 0.6)
    baseline_scores = (-0.5, 0.4, -0.1, 0.2, 0.3, -0.2, -0.3, 0.1)
    run_id = "1" * 32
    cnn_rows = []
    baseline_rows = []
    for item, cnn_score, baseline_score in zip(manifest, cnn_scores, baseline_scores):
        kept = item.source_speaker != "speaker-7"
        cnn_rows.append(
            external_run.ExternalScoreRow(
                item.file,
                0,
                item.source_speaker,
                None,
                item.label,
                cnn_score,
                0.8 if item.label == "FAKE" else 0.2,
                "synthetic" if item.label == "FAKE" else "authentic",
                kept,
                "bundle-v1",
                run_id,
            )
        )
        baseline_rows.append(
            external_run.ExternalScoreRow(
                item.file,
                0,
                item.source_speaker,
                None,
                item.label,
                baseline_score,
                None,
                "synthetic" if baseline_score >= 0 else "authentic",
                kept,
                None,
                run_id,
            )
        )
    manifest_sha = "d" * 64
    cnn_paths = external_run.write_external_score_file(
        tmp_path / "cnn",
        model_id="bundle-v1",
        model_role="cnn_bundle",
        model_artifact_sha256="a" * 64,
        manifest_sha256=manifest_sha,
        rows=cnn_rows,
    )
    baseline_paths = external_run.write_external_score_file(
        tmp_path / "baseline",
        model_id="baseline-v1",
        model_role="baseline",
        model_artifact_sha256="b" * 64,
        manifest_sha256=manifest_sha,
        rows=baseline_rows,
    )
    cnn_file = external_run.read_external_score_file(*cnn_paths)
    baseline_file = external_run.read_external_score_file(*baseline_paths)

    result = external_run.paired_eer_comparison(
        cnn_file,
        baseline_file,
        manifest,
        view="unfiltered",
        n_resamples=40,
    )
    assert result.n_windows == 8
    assert result.n_speakers == 8
    assert result.difference == pytest.approx(result.cnn_eer - result.baseline_eer)
    assert result.interval is not None
    assert result.interval.estimate == pytest.approx(result.difference)
    assert result.interval.n_resamples == 40
    assert result.n_invalid == result.interval.n_invalid

    filtered = external_run.paired_eer_comparison(
        cnn_file,
        baseline_file,
        manifest,
        view="filtered",
        n_resamples=40,
    )
    assert filtered.n_windows == 7
    assert filtered.n_speakers == 7


def test_paired_eer_comparison_refuses_cross_run_or_manifest_files(tmp_path):
    manifest = (
        external_run.ManifestFile("AUDIO/REAL/a.wav", "REAL", "speaker-a", None, 2.0),
        external_run.ManifestFile("AUDIO/FAKE/b.wav", "FAKE", "speaker-b", None, 2.0),
    )
    run_id = "1" * 32
    cnn = external_run.ExternalScoreRow(
        manifest[0].file,
        0,
        "speaker-a",
        None,
        "REAL",
        0.1,
        0.2,
        "authentic",
        True,
        "bundle-v1",
        run_id,
    )
    baseline = external_run.ExternalScoreRow(
        manifest[0].file,
        0,
        "speaker-a",
        None,
        "REAL",
        -0.1,
        None,
        "authentic",
        True,
        None,
        run_id,
    )
    cnn_file = (
        {"model_role": "cnn_bundle", "run_id": run_id, "manifest_sha256": "d" * 64},
        (cnn,),
    )
    baseline_file = (
        {"model_role": "baseline", "run_id": run_id, "manifest_sha256": "e" * 64},
        (baseline,),
    )
    with pytest.raises(ValueError, match="manifest_sha256"):
        external_run.paired_eer_comparison(
            cnn_file,
            baseline_file,
            manifest,
            view="unfiltered",
            n_resamples=2,
        )

    baseline_file = (
        {"model_role": "baseline", "run_id": "2" * 32, "manifest_sha256": "d" * 64},
        (baseline,),
    )
    with pytest.raises(ValueError, match="run_id"):
        external_run.paired_eer_comparison(
            cnn_file,
            baseline_file,
            manifest,
            view="unfiltered",
            n_resamples=2,
        )


def test_external_score_writer_rejects_duplicate_windows_and_bad_probabilities(
    tmp_path,
):
    base = {
        "file": "AUDIO/FAKE/a.wav",
        "window_index": 0,
        "source_speaker": "a",
        "target_speaker": None,
        "label": "FAKE",
        "raw_score": 0.2,
        "prediction": "authentic",
        "kept_by_filter": True,
        "bundle_version": "bundle-v1",
        "run_id": "1" * 32,
    }
    duplicate = external_run.ExternalScoreRow(**base, calibrated_p=0.2)
    with pytest.raises(ValueError, match="duplicate"):
        external_run.write_external_score_file(
            tmp_path,
            model_id="v1",
            model_role="cnn_bundle",
            model_artifact_sha256="a" * 64,
            manifest_sha256="c" * 64,
            rows=[duplicate, duplicate],
        )
    invalid = external_run.ExternalScoreRow(
        **{**base, "window_index": 1}, calibrated_p=1.2
    )
    with pytest.raises(ValueError, match="calibrated_p"):
        external_run.write_external_score_file(
            tmp_path,
            model_id="v1",
            model_role="cnn_bundle",
            model_artifact_sha256="a" * 64,
            manifest_sha256="c" * 64,
            rows=[invalid],
        )


def test_rows_from_window_run_uses_policy_for_cnn_and_forced_binary_for_baseline():
    windows = external_run.WindowRun(
        (
            external_run.WindowScore(0, 0.5, 0.9, 0, True),
            external_run.WindowScore(1, 0.49, 0.9, 0, False),
        ),
        n_dropped_samples=0,
    )

    class Policy:
        def decide(self, scores):
            return (
                SimpleNamespace(p=0.5, prediction="uncertain"),
                SimpleNamespace(p=0.49, prediction="authentic"),
            )

    cnn = external_run.rows_from_window_run(
        windows,
        file="AUDIO/FAKE/a.wav",
        source_speaker="source",
        target_speaker="target",
        label="FAKE",
        run_id="1" * 32,
        model_role="cnn_bundle",
        bundle_version="bundle-v1",
        policy=Policy(),
    )
    baseline = external_run.rows_from_window_run(
        windows,
        file="AUDIO/FAKE/a.wav",
        source_speaker="source",
        target_speaker="target",
        label="FAKE",
        run_id="1" * 32,
        model_role="baseline",
        baseline_threshold=0.5,
    )
    assert [(row.calibrated_p, row.prediction) for row in cnn] == [
        (0.5, "uncertain"),
        (0.49, "authentic"),
    ]
    assert [(row.calibrated_p, row.prediction) for row in baseline] == [
        (None, "synthetic"),
        (None, "authentic"),
    ]
    assert tuple(row.kept_by_filter for row in cnn) == (True, False)


def test_rows_from_window_run_rejects_missing_policy_or_bad_decision_count():
    windows = external_run.WindowRun((external_run.WindowScore(0, 0.2, 1, 0, True),), 0)
    with pytest.raises(ValueError, match="risk policy"):
        external_run.rows_from_window_run(
            windows,
            file="AUDIO/REAL/a.wav",
            source_speaker="a",
            target_speaker=None,
            label="REAL",
            run_id="1" * 32,
            model_role="cnn_bundle",
            bundle_version="bundle-v1",
        )

    class EmptyPolicy:
        def decide(self, scores):
            return ()

    with pytest.raises(ValueError, match="different number"):
        external_run.rows_from_window_run(
            windows,
            file="AUDIO/REAL/a.wav",
            source_speaker="a",
            target_speaker=None,
            label="REAL",
            run_id="1" * 32,
            model_role="cnn_bundle",
            bundle_version="bundle-v1",
            policy=EmptyPolicy(),
        )


def test_file_level_views_average_probabilities_after_filter_without_rescoring():
    manifest = [
        external_run.ManifestFile("AUDIO/REAL/a.wav", "REAL", "speaker-a", None, 4.0),
        external_run.ManifestFile(
            "AUDIO/FAKE/b.wav", "FAKE", "speaker-b", "target-b", 2.0
        ),
        external_run.ManifestFile(
            "AUDIO/FAKE/c.wav", "FAKE", "speaker-b", "target-c", 2.0
        ),
    ]

    def row(file, window, label, speaker, target, p, kept):
        return external_run.ExternalScoreRow(
            file=file,
            window_index=window,
            source_speaker=speaker,
            target_speaker=target,
            label=label,
            raw_score=p,
            calibrated_p=p,
            prediction="authentic",
            kept_by_filter=kept,
            bundle_version="bundle-v1",
            run_id="1" * 32,
        )

    scores = [
        row("AUDIO/REAL/a.wav", 0, "REAL", "speaker-a", None, 0.2, True),
        row("AUDIO/REAL/a.wav", 1, "REAL", "speaker-a", None, 0.8, False),
        row("AUDIO/FAKE/b.wav", 0, "FAKE", "speaker-b", "target-b", 0.9, True),
        row("AUDIO/FAKE/c.wav", 0, "FAKE", "speaker-b", "target-c", 0.1, False),
    ]
    filtered = external_run.file_level_view(
        scores, manifest, threshold=0.5, half_width=0.1, view="filtered", n_resamples=20
    )
    unfiltered = external_run.file_level_view(
        scores,
        manifest,
        threshold=0.5,
        half_width=0.1,
        view="unfiltered",
        n_resamples=20,
    )
    filtered_by_file = {row.file: row for row in filtered.rows}
    unfiltered_by_file = {row.file: row for row in unfiltered.rows}
    assert filtered_by_file["AUDIO/REAL/a.wav"].mean_p == pytest.approx(0.2)
    assert filtered_by_file["AUDIO/REAL/a.wav"].prediction == "authentic"
    assert unfiltered_by_file["AUDIO/REAL/a.wav"].mean_p == pytest.approx(0.5)
    assert unfiltered_by_file["AUDIO/REAL/a.wav"].prediction == "uncertain"
    assert filtered_by_file["AUDIO/FAKE/c.wav"].mean_p is None
    assert filtered_by_file["AUDIO/FAKE/c.wav"].prediction is None
    assert filtered.counts_by_class["FAKE"] == {
        "expected": 2,
        "evaluable": 1,
        "no_window": 1,
    }
    assert filtered.metrics["n_files"] == 2
    assert filtered.metrics["eer"] is not None
    assert filtered.metrics["n_speakers"] == 2
    assert filtered.metrics["intervals"]["eer"]["low"] is not None
    assert filtered.metrics["intervals"]["eer"]["n_resamples"] == 20
    assert unfiltered.counts_by_class["FAKE"]["evaluable"] == 2


def test_file_level_view_rejects_missing_or_duplicate_manifest_windows():
    manifest = [external_run.ManifestFile("AUDIO/REAL/a.wav", "REAL", "a", None, 4.0)]
    one_row = [
        external_run.ExternalScoreRow(
            file="AUDIO/REAL/a.wav",
            window_index=0,
            source_speaker="a",
            target_speaker=None,
            label="REAL",
            raw_score=0.1,
            calibrated_p=0.1,
            prediction="authentic",
            kept_by_filter=True,
            bundle_version="bundle-v1",
            run_id="1" * 32,
        )
    ]
    with pytest.raises(ValueError, match="incomplete"):
        external_run.file_level_view(
            one_row, manifest, threshold=0.5, half_width=0.1, view="unfiltered"
        )
    with pytest.raises(ValueError, match="duplicate"):
        external_run.file_level_view(
            one_row + one_row,
            manifest,
            threshold=0.5,
            half_width=0.1,
            view="unfiltered",
        )


def test_speaker_level_view_keeps_all_eight_speakers_and_propagates_undefined_values():
    manifest = []
    scores = []
    for index in range(8):
        speaker = f"speaker-{index}"
        real_file = f"AUDIO/REAL/{speaker}.wav"
        fake_file = f"AUDIO/FAKE/{speaker}-to-target.wav"
        manifest.extend(
            (
                external_run.ManifestFile(real_file, "REAL", speaker, None, 2.0),
                external_run.ManifestFile(fake_file, "FAKE", speaker, "target", 2.0),
            )
        )
        kept = index < 7
        scores.extend(
            (
                external_run.ExternalScoreRow(
                    file=real_file,
                    window_index=0,
                    source_speaker=speaker,
                    target_speaker=None,
                    label="REAL",
                    raw_score=0.1,
                    calibrated_p=0.1,
                    prediction="authentic",
                    kept_by_filter=kept,
                    bundle_version="bundle-v1",
                    run_id="1" * 32,
                ),
                external_run.ExternalScoreRow(
                    file=fake_file,
                    window_index=0,
                    source_speaker=speaker,
                    target_speaker="target",
                    label="FAKE",
                    raw_score=0.9,
                    calibrated_p=0.9,
                    prediction="synthetic",
                    kept_by_filter=kept,
                    bundle_version="bundle-v1",
                    run_id="1" * 32,
                ),
            )
        )
    report = external_run.speaker_level_view(
        scores,
        manifest,
        model_role="cnn_bundle",
        threshold=0.5,
        half_width=0.1,
        view="filtered",
    )
    assert len(report.rows) == report.averages["eer"].n_speakers == 8
    assert report.averages["eer"].n_defined == 7
    assert report.averages["eer"].undefined_speakers == ("speaker-7",)
    assert (
        report.undefined_reasons["eer"]["speaker-7"]
        == "no windows retained in this view"
    )
    assert report.rows[-1].n == 0
    assert report.filter_counts_by_class["REAL"] == {
        "complete_windows": 8,
        "kept": 7,
        "dropped": 1,
        "retention_rate": 0.875,
    }
    assert report.filter_counts_by_speaker["speaker-7"]["retention_rate"] == 0.0


def test_speaker_level_view_rejects_roster_not_equal_to_eight():
    manifest = [external_run.ManifestFile("AUDIO/REAL/a.wav", "REAL", "a", None, 2.0)]
    row = external_run.ExternalScoreRow(
        file="AUDIO/REAL/a.wav",
        window_index=0,
        source_speaker="a",
        target_speaker=None,
        label="REAL",
        raw_score=0.1,
        calibrated_p=0.1,
        prediction="authentic",
        kept_by_filter=True,
        bundle_version="bundle-v1",
        run_id="1" * 32,
    )
    with pytest.raises(ValueError, match="eight manifest source speakers"):
        external_run.speaker_level_view(
            [row],
            manifest,
            model_role="cnn_bundle",
            threshold=0.5,
            half_width=0.1,
            view="unfiltered",
        )


def test_strict_report_json_converts_undefined_values_and_is_immutable(tmp_path):
    report = {
        "summary": external_run.PairedEERView(
            "filtered",
            0,
            0,
            None,
            None,
            None,
            None,
            2000,
            0,
            "no windows retained in this view",
        ),
        "nan": float("nan"),
        "numpy_nan": np.float32("nan"),
        "path": tmp_path / "scores.csv",
    }
    path, digest = external_run.write_external_report(tmp_path / "report.json", report)
    decoded = json.loads(path.read_text(encoding="utf-8"))
    assert decoded["nan"] is None
    assert decoded["numpy_nan"] is None
    assert decoded["summary"]["difference"] is None
    assert decoded["summary"]["undefined_reason"] == "no windows retained in this view"
    assert decoded["path"] == str(tmp_path / "scores.csv")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        external_run.write_external_report(path, report)
