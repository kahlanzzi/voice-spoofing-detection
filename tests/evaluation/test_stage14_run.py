"""Stage 14 claim/score lifecycle tests; synthetic rows only, no dataset access."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.evaluation.stage14_run as run
from src.evaluation.clip_scores import ClipScores
from src.evaluation.run_registry import COMPLETED, STARTED, RunRegistry
from src.evaluation.robustness import CONDITION_IDS, RobustnessError, load_robustness_config


COMMIT = "a" * 40
REPO = Path(__file__).resolve().parents[2]


def _clean(model_kind: str) -> ClipScores:
    baseline = model_kind == run.MODEL_BASELINE
    return ClipScores(
        metadata={
            "evaluation_purpose": "stage11_test",
            "run_id": "b" * 32,
            "model_id": "baseline-v1" if baseline else "cnn-v1-seed0",
            "model_role": run.ROLE_BASELINE if baseline else run.ROLE_BUNDLE,
            "model_artifact_sha256": "c" * 64,
            "manifest_sha256": "d" * 64,
            "n_rows": 2,
            "csv_sha256": "e" * 64,
            "applicable_columns": [],
        },
        clip_id=("FAKE/a.wav", "REAL/b.wav"),
        label=("FAKE", "REAL"),
        group_id=("g1", "g2"),
        raw_score=np.asarray([0.9, 0.1], dtype=np.float64),
        p=None if baseline else np.asarray([0.9, 0.1], dtype=np.float64),
        prediction=("synthetic", "authentic"),
    )


def _context(model_kind: str) -> run.Context:
    baseline = model_kind == run.MODEL_BASELINE
    rows = pd.DataFrame({
        "path": ["FAKE/a.wav", "REAL/b.wav"],
        "label": ["FAKE", "REAL"],
        "group": ["g1", "g2"],
        "subgroup": ["FAKE_other", "REAL"],
    })
    return run.Context(
        commit=COMMIT,
        model_kind=model_kind,
        model_id="baseline-v1" if baseline else "cnn-v1-seed0",
        model_role=run.ROLE_BASELINE if baseline else run.ROLE_BUNDLE,
        artifact_sha256="c" * 64,
        profile=None,
        manifest=None,
        rows=rows,
        clean=_clean(model_kind),
        robustness=load_robustness_config(REPO / "configs" / "robustness.yaml"),
        codec=None,
        checks={"synthetic": {"expected": True, "actual": True, "passed": True}},
        f2_manifest_sha256="d" * 64,
        profile_config_sha256="f" * 64,
        baseline=object() if baseline else None,
        bundle=object() if not baseline else None,
        batch_size=None if baseline else 2,
    )


@pytest.fixture
def lifecycle(monkeypatch):
    monkeypatch.setattr(run, "code_version", lambda: {"git_commit": COMMIT})
    monkeypatch.setattr(run, "environment_record", lambda: ({"runtime": "synthetic"}, "x==1\n"))
    monkeypatch.setattr(
        run, "preflight",
        lambda checks, registry_path, model_kind, **_kwargs: _context(model_kind),
    )


def _claim(tmp_path, model_kind: str):
    registry = tmp_path / "registry.jsonl"
    out = tmp_path / model_kind
    record = run.claim(
        registry_path=registry,
        out_dir=out,
        results_ref=f"stage14/{model_kind}/run_report.json",
        model_kind=model_kind,
    )
    return registry, out, record


def test_check_is_read_only_and_covers_the_complete_grid(tmp_path, lifecycle):
    registry = tmp_path / "registry.jsonl"
    report = run.check(registry_path=registry, model_kind=run.MODEL_BASELINE)
    assert report["model_id"] == "baseline-v1"
    assert report["conditions"] == list(CONDITION_IDS)
    assert report["counts"] == {
        "label": {"FAKE": 1, "REAL": 1},
        "subgroup": {"FAKE_other": 1, "REAL": 1},
    }
    assert not registry.exists()
    assert list(tmp_path.iterdir()) == []


def test_claims_are_separate_single_model_sets_and_durable(tmp_path, lifecycle):
    registry, baseline_out, baseline = _claim(tmp_path, run.MODEL_BASELINE)
    bundle_out = tmp_path / run.MODEL_BUNDLE
    bundle = run.claim(
        registry_path=registry,
        out_dir=bundle_out,
        results_ref="stage14/cnn_bundle/run_report.json",
        model_kind=run.MODEL_BUNDLE,
    )

    records = RunRegistry(registry).records()
    assert [(item.status, item.model_set) for item in records] == [
        (STARTED, ("baseline-v1",)),
        (STARTED, ("cnn-v1-seed0",)),
    ]
    assert baseline["conditions"] == list(CONDITION_IDS)
    assert bundle["conditions"] == list(CONDITION_IDS)
    for out in (baseline_out, bundle_out):
        assert (out / run.CLAIM_JSON).is_file()
        assert (out / run.ENVIRONMENT_JSON).is_file()
        assert (out / run.PIP_FREEZE).is_file()


def test_claim_refuses_a_second_run_for_the_same_model(tmp_path, lifecycle):
    registry, _, first = _claim(tmp_path, run.MODEL_BASELINE)
    with pytest.raises(RobustnessError, match="second run needs a new decision"):
        run.claim(
            registry_path=registry,
            out_dir=tmp_path / "again",
            results_ref="stage14/again/run_report.json",
            model_kind=run.MODEL_BASELINE,
        )
    assert len(RunRegistry(registry).records()) == 1
    assert first["run_id"] == RunRegistry(registry).records()[0].run_id


def test_score_writes_all_nine_conditions_and_completes_once(tmp_path, lifecycle, monkeypatch):
    registry, out, claimed = _claim(tmp_path, run.MODEL_BASELINE)

    monkeypatch.setattr(
        run,
        "_condition_features",
        lambda _root, _ctx, condition: (
            np.zeros((2, 40), dtype=np.float32),
            {
                "feature": "mfcc",
                "array_sha256": condition.encode().hex().ljust(64, "0")[:64],
                "features_sha256": "1" * 64,
                "degraded_waveforms_sha256": "2" * 64,
                "n_clips": 2,
            },
        ),
    )
    monkeypatch.setattr(
        run,
        "_score_condition",
        lambda _ctx, _features: (
            np.asarray([0.9, 0.1], dtype=np.float64),
            None,
            ("synthetic", "authentic"),
        ),
    )
    monkeypatch.setattr(
        run,
        "compare_condition",
        lambda _clean, _degraded, *, condition_id, config: {"condition": condition_id},
    )

    report = run.score(
        root=tmp_path / "never-read",
        registry_path=registry,
        out_dir=out,
        run_id=claimed["run_id"],
        model_kind=run.MODEL_BASELINE,
    )

    assert list(report["conditions"]) == list(CONDITION_IDS)
    assert report["scoring"] == {"device": "cpu", "passes": 2}
    assert (out / run.METRICS_REPORT).is_file()
    assert (out / run.RUN_REPORT).is_file()
    record = RunRegistry(registry).records()[-1]
    assert record.status == COMPLETED
    assert record.run_id == claimed["run_id"]
    with pytest.raises(RobustnessError, match="already started scoring"):
        run.score(
            root=tmp_path,
            registry_path=registry,
            out_dir=out,
            run_id=claimed["run_id"],
            model_kind=run.MODEL_BASELINE,
        )


def test_wrong_run_id_is_refused_before_the_once_only_marker(tmp_path, lifecycle):
    registry, out, _ = _claim(tmp_path, run.MODEL_BASELINE)
    with pytest.raises(RobustnessError, match="claim run_id/commit mismatch"):
        run.score(
            root=tmp_path,
            registry_path=registry,
            out_dir=out,
            run_id="f" * 32,
            model_kind=run.MODEL_BASELINE,
        )
    assert not (out / run.SCORING_MARKER).exists()
