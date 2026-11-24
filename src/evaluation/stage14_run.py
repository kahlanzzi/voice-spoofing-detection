"""Official Stage 14 v1 FoR robustness runner (Protocol section 9; D3, D41, D58).

The runner has one claim/score lifecycle for ``baseline-v1`` and another for the F4 CNN bundle
seed. Each claim covers the complete nine-condition grid. Clean scores come from the completed
Stage 11 run; clean audio is never scored again. A score marker is created before the first FoR
test read, and a failed or interrupted claim cannot be retried without a new decision.

No function accepts a split or dataset argument. FoR/test and the closed
``stage14_robustness`` purpose are constants. DEEP-VOICE has no path through this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.augmentation.codec import FFmpegCodecBackend
from src.augmentation.config import load_augmentation_config
from src.augmentation.selection_set_build import official_runtime_record
from src.bundle.f4 import expected_from_evidence as f4_expected_from_evidence
from src.bundle.f4 import verify as verify_f4
from src.bundle.model_loader import load_bundle_model
from src.evaluation.clip_scores import ClipScores, read_clip_scores, write_clip_scores
from src.evaluation.robustness import (
    CONDITION_IDS,
    RobustnessConfig,
    RobustnessError,
    apply_condition,
    compare_condition,
    load_robustness_config,
)
from src.evaluation.run_registry import COMPLETED, STARTED, RunRegistry
from src.evaluation.stage11_run import (
    BASELINE_ARTIFACT_SHA256,
    BASELINE_ID,
    CLAIM_JSON,
    DEVICE,
    ENVIRONMENT_JSON,
    ENVIRONMENT_SCORE_JSON,
    LOG_MEL,
    MFCC,
    MODEL_IDS,
    PIP_FREEZE,
    SCORING_MARKER,
    _check_results_ref,
    _counts,
    _create,
    _json_text,
    array_sha256,
    baseline_outputs,
    bundle_outputs,
    environment_gate,
    environment_record,
    features_sha256,
    load_evidence,
    reproduction_gate,
    split_rows,
)
from src.models.baseline import artifact_sha256
from src.models.baseline import load as load_baseline
from src.preprocessing.features import extract_features
from src.preprocessing.profile import load_profile
from src.preprocessing.validation import (
    F2_MANIFEST_SHA256,
    code_version,
    load_f2_manifest,
    read_verified_audio,
    sha256_file,
)
from src.tuning.risk_run import score_clips


SCHEMA_VERSION = 1
DATASET = "for"
SPLIT = "test"
PURPOSE = "stage14_robustness"
MODEL_BASELINE = "baseline"
MODEL_BUNDLE = "cnn_bundle"
MODEL_KINDS = (MODEL_BASELINE, MODEL_BUNDLE)
TIMING_CONDITION = "low_pass_3400hz"
ROLE_BASELINE = "baseline"
ROLE_BUNDLE = "cnn_bundle_seed"
RUN_REPORT = "run_report.json"
METRICS_REPORT = "metrics_report.json"
CONDITIONS_DIR = "conditions"
CLIP_SCORES_DIR = "clip_scores"
ROBUSTNESS_CONFIG = "configs/robustness.yaml"
F4_RECORD_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs" / "audit_evidence" / "stage12_bundle" / "f4_record.json"
)
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
APPROVAL_SCOPE = "stage14_v1_official"
H6_DEFINITION = "forced_binary_false_synthetic_rate_at_frozen_threshold"
REQUIRED_APPROVERS = frozenset({"Donia", "Iheb", "Khalil", "Malak"})


@dataclass
class Context:
    commit: str
    model_kind: str
    model_id: str
    model_role: str
    artifact_sha256: str
    profile: Any
    manifest: Any
    rows: Any
    clean: ClipScores
    robustness: RobustnessConfig
    codec: FFmpegCodecBackend
    checks: dict[str, Any]
    f2_manifest_sha256: str
    profile_config_sha256: str
    baseline: Any = None
    bundle: Any = None
    batch_size: int | None = None


def _check(checks: dict, key: str, expected, actual) -> None:
    same = type(expected) is type(actual) and expected == actual
    checks[key] = {"expected": expected, "actual": actual, "passed": same}
    if not same:
        raise RobustnessError(f"check failed: {key}: expected {expected!r}, got {actual!r}")


def _record(checks: dict, key: str, actual) -> None:
    checks[key] = {"expected": "recorded", "actual": actual, "passed": True}


def _model_kind(value: str) -> str:
    if value not in MODEL_KINDS:
        raise RobustnessError(f"model_kind must be one of {MODEL_KINDS}, got {value!r}")
    return value


def validate_official_approval(path: str | Path, commit: str) -> dict[str, Any]:
    """Bind an irreversible claim to the reviewed commit and explicit four-person approval."""
    path = Path(path)
    try:
        record = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RobustnessError(f"cannot read official approval record {path}: {error}") from None
    expected_keys = {
        "schema_version", "scope", "approved_commit", "h6_definition", "approvals"}
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise RobustnessError(
            f"official approval record keys must be exactly {sorted(expected_keys)}")
    if (record["schema_version"], record["scope"], record["approved_commit"],
            record["h6_definition"]) != (1, APPROVAL_SCOPE, commit, H6_DEFINITION):
        raise RobustnessError(
            "official approval record does not bind this commit and the reviewed H6 definition")
    approvals = record["approvals"]
    if not isinstance(approvals, dict) or set(approvals) != REQUIRED_APPROVERS:
        raise RobustnessError(
            f"official approval requires exactly {sorted(REQUIRED_APPROVERS)}")
    if any(not isinstance(value, str) or not value.strip() for value in approvals.values()):
        raise RobustnessError("every official approval must contain a non-empty evidence reference")
    return record


def read_stage11_clean_scores(
    score_dir: str | Path,
    *,
    model_id: str,
    registry_path: str | Path,
    manifest_path: str | Path,
    run_report_path: str | Path,
) -> ClipScores:
    """Bind reused clean scores to F2 and exactly one completed Stage 11 registry run (D41)."""
    actual_manifest = sha256_file(Path(manifest_path))
    if actual_manifest != F2_MANIFEST_SHA256:
        raise RobustnessError(
            f"F2 manifest SHA-256 mismatch: expected {F2_MANIFEST_SHA256}, got {actual_manifest}")
    scores = read_clip_scores(score_dir, model_id)
    meta = scores.metadata
    if meta["evaluation_purpose"] != "stage11_test":
        raise RobustnessError(
            f"clean score purpose must be 'stage11_test', got {meta['evaluation_purpose']!r}")
    if meta["manifest_sha256"] != F2_MANIFEST_SHA256:
        raise RobustnessError(
            f"clean score manifest SHA-256 {meta['manifest_sha256']} differs from F2 "
            f"{F2_MANIFEST_SHA256}")
    if meta["model_role"] not in {ROLE_BASELINE, ROLE_BUNDLE}:
        raise RobustnessError(
            "Stage 14 v1 accepts only baseline or frozen CNN bundle clean scores, got "
            f"{meta['model_role']!r}")
    matches = [
        record for record in RunRegistry(registry_path).records()
        if record.status == COMPLETED
        and record.run_id == meta["run_id"]
        and record.dataset == DATASET
        and record.split == SPLIT
        and record.evaluation_purpose == "stage11_test"
        and record.model_set == MODEL_IDS
        and model_id in record.model_set
    ]
    if len(matches) != 1:
        raise RobustnessError(
            f"clean score run_id {meta['run_id']} has {len(matches)} matching completed "
            "Stage 11 registry records; expected exactly one")
    try:
        report = json.loads(Path(run_report_path).read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RobustnessError(f"cannot read Stage 11 run report {run_report_path}: {error}") from None
    expected_identity = (1, DATASET, SPLIT, "stage11_test", meta["run_id"], matches[0].code_commit)
    actual_identity = tuple(report.get(key) for key in (
        "schema_version", "dataset", "split", "evaluation_purpose", "run_id", "code_commit"))
    if actual_identity != expected_identity:
        raise RobustnessError(
            f"Stage 11 run report identity differs from the completed registry record: "
            f"{actual_identity!r} != {expected_identity!r}")
    model_set = report.get("model_set")
    if not isinstance(model_set, dict) or tuple(sorted(model_set)) != MODEL_IDS:
        raise RobustnessError("Stage 11 run report does not contain the exact v1 model set")
    files = report.get("clip_scores", {}).get("files", {})
    expected_file = files.get(model_id)
    if not isinstance(expected_file, dict):
        raise RobustnessError(f"Stage 11 run report has no clip-score record for {model_id}")
    score_dir = Path(score_dir)
    actual_files = {
        "csv_sha256": sha256_file(score_dir / f"{model_id}.csv"),
        "sidecar_sha256": sha256_file(score_dir / f"{model_id}.json"),
        "n_rows": len(scores.clip_id),
    }
    if expected_file != actual_files:
        raise RobustnessError(
            f"Stage 11 run report hashes/count for {model_id} differ from the clean score files")
    model_record = model_set[model_id]
    if not isinstance(model_record, dict) or (
        model_record.get("role"), model_record.get("artifact_sha256")
    ) != (meta["model_role"], meta["model_artifact_sha256"]):
        raise RobustnessError(
            f"Stage 11 run report model identity for {model_id} differs from its score sidecar")
    return scores


def _check_clean_rows(clean: ClipScores, rows) -> None:
    expected = (
        tuple(rows["path"].tolist()), tuple(rows["label"].tolist()), tuple(rows["group"].tolist()))
    actual = (clean.clip_id, clean.label, clean.group_id)
    if actual != expected:
        names = ("clip_id", "label", "group_id")
        changed = [name for name, left, right in zip(names, actual, expected) if left != right]
        raise RobustnessError(f"Stage 11 clean scores differ from F2 rows: {changed}")


def _base_preflight(
    checks: dict,
    *,
    model_kind,
    registry_path,
    clean_scores,
    stage11_run_report,
    profile_path,
    manifest_path,
    robustness_config,
    augmentation_config,
    colab_runtime_version,
    stage7_evidence,
    stage8_evidence,
    stage10_evidence,
):
    model_kind = _model_kind(model_kind)
    commit = code_version()["git_commit"]
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise RobustnessError(f"code commit must be full lowercase SHA-1, got {commit!r}")
    _record(checks, "code_commit", commit)
    environment_gate(checks)
    evidence = load_evidence(stage7_evidence, stage8_evidence, stage10_evidence)

    profile_path, manifest_path = Path(profile_path), Path(manifest_path)
    profile = load_profile(profile_path)
    profile_sha = sha256_file(profile_path)
    manifest_sha = sha256_file(manifest_path)
    _check(checks, "profile_fingerprint", evidence.profile_fingerprint, profile.fingerprint())
    _check(checks, "profile_config_sha256", evidence.profile_config_sha256, profile_sha)
    _check(checks, "f2_manifest_sha256", evidence.f2_manifest_sha256, manifest_sha)
    _check(checks, "f2_manifest_sha256_approved", F2_MANIFEST_SHA256, manifest_sha)
    manifest = load_f2_manifest(manifest_path)
    rows = split_rows(manifest, SPLIT)

    robust = load_robustness_config(robustness_config)
    augmentation = load_augmentation_config(Path(augmentation_config))
    _check(checks, "sample_rate", profile.sample_rate, robust.sample_rate)
    if not augmentation.ffmpeg_pin_complete:
        raise RobustnessError("FFmpeg pin is incomplete")
    codec = FFmpegCodecBackend(augmentation.ffmpeg, robust.sample_rate)
    codec.verify_pin()
    _record(checks, "ffmpeg_pin", augmentation.ffmpeg["expected_version"])
    runtime = official_runtime_record(codec, colab_runtime_version)
    _record(checks, "official_runtime", runtime)

    bundle_id = f"cnn-v1-seed{evidence.bundle_seed}"
    model_id = BASELINE_ID if model_kind == MODEL_BASELINE else bundle_id
    clean = read_stage11_clean_scores(
        clean_scores, model_id=model_id, registry_path=registry_path, manifest_path=manifest_path,
        run_report_path=stage11_run_report)
    expected_role = ROLE_BASELINE if model_kind == MODEL_BASELINE else ROLE_BUNDLE
    _check(checks, "clean.model_role", expected_role, clean.metadata["model_role"])
    _check_clean_rows(clean, rows)
    return (commit, model_kind, model_id, evidence, profile, profile_sha, manifest, rows, clean,
            robust, codec, manifest_sha)


def preflight(
    checks: dict,
    *,
    model_kind,
    registry_path,
    clean_scores,
    profile_path,
    manifest_path,
    robustness_config,
    augmentation_config,
    colab_runtime_version,
    stage7_evidence,
    stage8_evidence,
    stage10_evidence,
    baseline_dir=None,
    f4_bundle=None,
    f4_record=None,
    cache_a=None,
    cache_b=None,
    mfcc_reference=None,
) -> Context:
    """All no-test-audio checks. Baseline includes the D58 MFCC reproduction gate."""
    values = _base_preflight(
        checks, model_kind=model_kind, registry_path=registry_path, clean_scores=clean_scores,
        profile_path=profile_path, manifest_path=manifest_path,
        robustness_config=robustness_config, augmentation_config=augmentation_config,
        colab_runtime_version=colab_runtime_version,
        stage7_evidence=stage7_evidence, stage8_evidence=stage8_evidence,
        stage10_evidence=stage10_evidence)
    (commit, model_kind, model_id, evidence, profile, profile_sha, manifest, rows, clean,
     robust, codec, manifest_sha) = values
    if model_kind == MODEL_BASELINE:
        if any(value is None for value in (baseline_dir, cache_a, cache_b, mfcc_reference)):
            raise RobustnessError(
                "baseline needs --baseline, --cache-a, --cache-b and --mfcc-reference (D58)")
        _check(checks, "baseline.artifact_sha256", BASELINE_ARTIFACT_SHA256,
               artifact_sha256(baseline_dir))
        baseline = load_baseline(baseline_dir)
        _check(checks, "baseline.version", BASELINE_ID, baseline.version)
        _check(checks, "baseline.threshold", evidence.baseline_threshold, baseline.threshold)
        _check(checks, "clean.model_artifact_sha256", BASELINE_ARTIFACT_SHA256,
               clean.metadata["model_artifact_sha256"])
        reproduction_gate(
            cache_a, cache_b, mfcc_reference, profile, profile_sha, evidence, checks)
        return Context(
            commit, model_kind, model_id, ROLE_BASELINE, BASELINE_ARTIFACT_SHA256, profile,
            manifest, rows, clean, robust, codec, checks, manifest_sha, profile_sha,
            baseline=baseline)

    if any(value is None for value in (f4_bundle, f4_record)):
        raise RobustnessError("cnn_bundle needs --f4-bundle and --f4-record")
    _check(checks, "f4.record_sha256", sha256_file(F4_RECORD_PATH),
           sha256_file(Path(f4_record)))
    verification = verify_f4(f4_bundle, f4_record)
    bundle = load_bundle_model(f4_bundle)
    expected = f4_expected_from_evidence()
    checkpoint_sha = bundle.bundle.checkpoint_sha256
    _check(checks, "f4.manifest_sha256", verification["manifest_sha256"],
           bundle.bundle.manifest_sha256)
    _check(checks, "f4.bundle_seed", evidence.bundle_seed, bundle.metadata.get("seed"))
    _check(checks, "f4.checkpoint_sha256", evidence.checkpoint_sha256[evidence.bundle_seed],
           checkpoint_sha)
    _check(checks, "f4.evidence_checkpoint_sha256", expected["checkpoint"]["sha256"],
           checkpoint_sha)
    _check(checks, "f4.profile_sha256", expected["profile"]["sha256"],
           bundle.bundle.manifest.components["profile"].sha256)
    _check(checks, "f4.policy_sha256", expected["policy"]["sha256"],
           bundle.bundle.manifest.components["policy"].sha256)
    _check(checks, "clean.model_artifact_sha256", checkpoint_sha,
           clean.metadata["model_artifact_sha256"])
    if bundle.architecture != "cnn":
        raise RobustnessError(f"F4 must load the v1 CNN architecture, got {bundle.architecture!r}")
    return Context(
        commit, model_kind, model_id, ROLE_BUNDLE, checkpoint_sha, profile, manifest, rows, clean,
        robust, codec, checks, manifest_sha, profile_sha, bundle=bundle,
        batch_size=evidence.batch_size)


def _stage14_records(registry: RunRegistry, model_id: str) -> list:
    return [
        record for record in registry.records()
        if (record.dataset, record.split, record.evaluation_purpose, record.model_set)
        == (DATASET, SPLIT, PURPOSE, (model_id,))]


def check(*, registry_path, **inputs) -> dict[str, Any]:
    """Run every no-test-audio preflight and return its evidence without writing or claiming."""
    ctx = preflight({}, registry_path=registry_path, **inputs)
    return {
        "schema_version": SCHEMA_VERSION,
        "code_commit": ctx.commit,
        "dataset": DATASET,
        "split": SPLIT,
        "evaluation_purpose": PURPOSE,
        "model_kind": ctx.model_kind,
        "model_id": ctx.model_id,
        "model_role": ctx.model_role,
        "model_artifact_sha256": ctx.artifact_sha256,
        "conditions": list(CONDITION_IDS),
        "checks": ctx.checks,
        "counts": _counts(ctx.rows),
        "clean_scores": {
            "run_id": ctx.clean.metadata["run_id"],
            "csv_sha256": ctx.clean.metadata["csv_sha256"],
        },
    }


def time_validation_condition(*, root, registry_path, **inputs) -> dict[str, Any]:
    """Time one reviewed condition on val-A without reading FoR test audio or writing output."""
    ctx = preflight({}, registry_path=registry_path, **inputs)
    validation_rows = split_rows(ctx.manifest, "val-A")
    validation_ctx = replace(ctx, rows=validation_rows)
    started = time.perf_counter()
    features, feature_record = _condition_features(root, validation_ctx, TIMING_CONDITION)
    features_done = time.perf_counter()
    _score_condition(validation_ctx, features)
    scoring_done = time.perf_counter()
    feature_seconds = features_done - started
    scoring_seconds = scoring_done - features_done
    total_seconds = scoring_done - started
    return {
        "schema_version": SCHEMA_VERSION,
        "code_commit": ctx.commit,
        "dataset": DATASET,
        "split": "val-A",
        "evaluation_purpose": "stage14_check_timing",
        "model_kind": ctx.model_kind,
        "model_id": ctx.model_id,
        "condition_id": TIMING_CONDITION,
        "n_rows": len(validation_rows),
        "feature_record": feature_record,
        "feature_seconds": feature_seconds,
        "two_pass_score_seconds": scoring_seconds,
        "condition_total_seconds": total_seconds,
        "estimated_nine_condition_seconds": total_seconds * len(CONDITION_IDS),
    }


def claim(*, registry_path, out_dir, results_ref, approval_record, **inputs) -> dict[str, Any]:
    """Validate without test audio, make one purpose claim, then write durable evidence."""
    results_ref = _check_results_ref(results_ref)
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise RobustnessError(f"output folder already exists: {out_dir}")
    ctx = preflight({}, registry_path=registry_path, **inputs)
    approval = validate_official_approval(approval_record, ctx.commit)
    registry = RunRegistry(registry_path)
    earlier = _stage14_records(registry, ctx.model_id)
    if earlier:
        record = earlier[0]
        raise RobustnessError(
            f"the registry already holds a {record.status} {PURPOSE} record for {ctx.model_id} "
            f"(run_id {record.run_id}); a second run needs a new decision")
    environment, freeze = environment_record()
    run_id = registry.claim(
        DATASET, SPLIT, (ctx.model_id,), ctx.commit, evaluation_purpose=PURPOSE)
    last = RunRegistry(registry_path).records()[-1]
    expected_key = (DATASET, SPLIT, PURPOSE, (ctx.model_id,))
    if (last.run_id, last.status, last.code_commit, last.key) != (
            run_id, STARTED, ctx.commit, expected_key):
        raise RobustnessError(f"registry's last record is not this Stage 14 claim: {last}")

    out_dir.mkdir(parents=True)
    environment_bytes = _json_text(environment).encode("utf-8")
    freeze_bytes = freeze.encode("utf-8")
    _create(out_dir / ENVIRONMENT_JSON, environment_bytes)
    _create(out_dir / PIP_FREEZE, freeze_bytes)
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "code_commit": ctx.commit,
        "registry_sha256": sha256_file(Path(registry_path)),
        "results_ref": results_ref,
        "dataset": DATASET,
        "split": SPLIT,
        "evaluation_purpose": PURPOSE,
        "model_kind": ctx.model_kind,
        "model_id": ctx.model_id,
        "model_role": ctx.model_role,
        "model_artifact_sha256": ctx.artifact_sha256,
        "conditions": list(CONDITION_IDS),
        "pre_claim_checks": ctx.checks,
        "clean_scores": {
            "run_id": ctx.clean.metadata["run_id"],
            "csv_sha256": ctx.clean.metadata["csv_sha256"]},
        "environment_files": {
            ENVIRONMENT_JSON: hashlib.sha256(environment_bytes).hexdigest(),
            PIP_FREEZE: hashlib.sha256(freeze_bytes).hexdigest()},
        "counts": _counts(ctx.rows),
        "official_approval_sha256": sha256_file(Path(approval_record)),
        "official_approval": approval,
    }
    _create(out_dir / CLAIM_JSON, _json_text(record).encode("utf-8"))
    return record


def _read_claim(out_dir: Path, run_id: str, commit: str, registry_path) -> tuple[dict, str]:
    data = (out_dir / CLAIM_JSON).read_bytes()
    record = json.loads(data.decode("utf-8"))
    if (record.get("run_id"), record.get("code_commit")) != (run_id, commit):
        raise RobustnessError(f"{CLAIM_JSON} is not for run_id {run_id} at commit {commit}")
    _check_results_ref(record.get("results_ref"))
    actual_registry = sha256_file(Path(registry_path))
    if actual_registry != record.get("registry_sha256"):
        raise RobustnessError(
            f"registry changed since claim: {actual_registry} != {record.get('registry_sha256')}")
    expected_files = record.get("environment_files", {})
    if set(expected_files) != {ENVIRONMENT_JSON, PIP_FREEZE}:
        raise RobustnessError("claim does not bind both environment files")
    for name, expected in expected_files.items():
        if sha256_file(out_dir / name) != expected:
            raise RobustnessError(f"{name} changed since claim")
    return record, hashlib.sha256(data).hexdigest()


def _condition_features(root, ctx: Context, condition_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    feature = MFCC if ctx.model_kind == MODEL_BASELINE else LOG_MEL
    shape = (40,) if feature == MFCC else (1, 64, 197)
    values = np.empty((len(ctx.rows), *shape), dtype=np.float32)
    waveform_hash = hashlib.sha256()
    for index, (_, row) in enumerate(ctx.rows.iterrows()):
        waveform, sample_rate = read_verified_audio(root, row)
        if sample_rate != ctx.robustness.sample_rate:
            raise RobustnessError(
                f"{row['path']}: Stage 14 input must be {ctx.robustness.sample_rate} Hz, "
                f"got {sample_rate}")
        degraded = apply_condition(
            waveform, condition_id=condition_id, clip_id=str(row["path"]),
            config=ctx.robustness, codec_backend=ctx.codec)
        waveform_hash.update(np.ascontiguousarray(degraded, dtype="<f4").tobytes())
        _, log_mel, mfcc = extract_features(degraded, sample_rate, ctx.profile)
        values[index] = mfcc if feature == MFCC else log_mel
    digest = array_sha256(feature, values)
    labels = ctx.rows["label"].tolist()
    paths = ctx.rows["path"].tolist()
    return values, {
        "feature": feature,
        "array_sha256": digest,
        "features_sha256": features_sha256(paths, labels, {feature: digest}),
        "degraded_waveforms_sha256": waveform_hash.hexdigest(),
        "n_clips": len(values),
    }


def _score_condition(ctx: Context, features: np.ndarray):
    def once() -> np.ndarray:
        if ctx.model_kind == MODEL_BASELINE:
            return np.asarray(ctx.baseline.score(features), dtype=np.float64)
        return score_clips(
            ctx.bundle.model, ctx.bundle.mean, ctx.bundle.std, features, int(ctx.batch_size))

    first, second = once(), once()
    if first.dtype != np.float64 or first.shape != (len(ctx.rows),):
        raise RobustnessError(
            f"scores must be float64 ({len(ctx.rows)},), got {first.dtype} {first.shape}")
    if first.tobytes() != second.tobytes():
        raise RobustnessError("scores differ between two CPU passes")
    if ctx.model_kind == MODEL_BASELINE:
        return first, None, baseline_outputs(ctx.baseline, features, first)
    p, prediction = bundle_outputs(ctx.bundle.bundle.policy, first)
    return first, p, prediction


def score(*, root, registry_path, out_dir, run_id, approval_record, **inputs) -> dict[str, Any]:
    """Consume one claim exactly once and score all nine degraded conditions."""
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise RobustnessError(f"run_id must be 32 lowercase hex chars, got {run_id!r}")
    out_dir = Path(out_dir)
    claim_path = out_dir / CLAIM_JSON
    if not claim_path.is_file():
        raise RobustnessError(f"{claim_path} does not exist")
    claimed = json.loads(claim_path.read_bytes().decode("utf-8"))
    commit = code_version()["git_commit"]
    if (claimed.get("run_id"), claimed.get("code_commit")) != (run_id, commit):
        raise RobustnessError("claim run_id/commit mismatch; nothing was written")
    approval = validate_official_approval(approval_record, commit)
    if (claimed.get("official_approval_sha256") != sha256_file(Path(approval_record))
            or claimed.get("official_approval") != approval):
        raise RobustnessError("official approval record changed since the claim")

    marker = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        _create(out_dir / SCORING_MARKER, _json_text(marker).encode("utf-8"))
    except ValueError:
        raise RobustnessError(
            f"{SCORING_MARKER} exists: this Stage 14 claim already started scoring") from None
    environment_score_bytes = _json_text(environment_record()[0]).encode("utf-8")
    _create(out_dir / ENVIRONMENT_SCORE_JSON, environment_score_bytes)

    ctx = preflight({}, registry_path=registry_path, **inputs)
    if (claimed.get("model_kind"), claimed.get("model_id")) != (ctx.model_kind, ctx.model_id):
        raise RobustnessError("claim model differs from scoring inputs")
    records = _stage14_records(RunRegistry(registry_path), ctx.model_id)
    if len(records) != 1 or (
        records[0].run_id, records[0].status, records[0].code_commit
    ) != (run_id, STARTED, ctx.commit):
        raise RobustnessError("registry does not hold exactly this started Stage 14 claim")
    claim_record, claim_sha = _read_claim(out_dir, run_id, ctx.commit, registry_path)
    if _counts(ctx.rows) != claim_record["counts"]:
        raise RobustnessError("F2 row counts changed since claim")

    condition_root = out_dir / CONDITIONS_DIR
    condition_root.mkdir()
    columns = {
        "clip_id": ctx.rows["path"].tolist(),
        "label": ctx.rows["label"].tolist(),
        "group_id": ctx.rows["group"].tolist()}
    condition_records: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    for condition_id in CONDITION_IDS:
        features, feature_record = _condition_features(root, ctx, condition_id)
        raw, p, prediction = _score_condition(ctx, features)
        folder = condition_root / condition_id / CLIP_SCORES_DIR
        folder.mkdir(parents=True)
        csv_path, sidecar_path = write_clip_scores(
            folder, evaluation_purpose=PURPOSE, run_id=run_id, model_id=ctx.model_id,
            model_role=ctx.model_role, model_artifact_sha256=ctx.artifact_sha256,
            manifest_sha256=ctx.f2_manifest_sha256, raw_score=raw, p=p,
            prediction=prediction, **columns)
        degraded = read_clip_scores(folder, ctx.model_id)
        forced_binary_threshold = (
            float(ctx.bundle.bundle.policy.threshold)
            if ctx.model_kind == MODEL_BUNDLE else None
        )
        metrics[condition_id] = compare_condition(
            ctx.clean, degraded, condition_id=condition_id, config=ctx.robustness,
            forced_binary_threshold=forced_binary_threshold)
        condition_records[condition_id] = {
            "features": feature_record,
            "csv_sha256": sha256_file(csv_path),
            "sidecar_sha256": sha256_file(sidecar_path),
            "n_rows": len(ctx.rows)}

    metrics_payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model_id": ctx.model_id,
        "conditions": metrics}
    _create(out_dir / METRICS_REPORT, _json_text(metrics_payload).encode("utf-8"))
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "code_commit": ctx.commit,
        "dataset": DATASET,
        "split": SPLIT,
        "evaluation_purpose": PURPOSE,
        "model_kind": ctx.model_kind,
        "model_id": ctx.model_id,
        "model_role": ctx.model_role,
        "model_artifact_sha256": ctx.artifact_sha256,
        "claim_sha256": claim_sha,
        "scoring_marker_sha256": sha256_file(out_dir / SCORING_MARKER),
        "pre_scoring_checks": ctx.checks,
        "environment_files": claim_record["environment_files"],
        "environment_score_sha256": hashlib.sha256(environment_score_bytes).hexdigest(),
        "environment_same_as_claim": environment_score_bytes == (
            out_dir / ENVIRONMENT_JSON).read_bytes(),
        "scoring": {"device": DEVICE, "passes": 2},
        "clean_scores": claim_record["clean_scores"],
        "conditions": condition_records,
        "metrics_report_sha256": sha256_file(out_dir / METRICS_REPORT),
        "counts": claim_record["counts"]}
    _create(out_dir / RUN_REPORT, _json_text(report).encode("utf-8"))
    RunRegistry(registry_path).complete(run_id, results_ref=claim_record["results_ref"])
    return report


def _inputs(args) -> dict[str, Any]:
    return {
        "model_kind": args.model_kind,
        "clean_scores": args.clean_scores,
        "stage11_run_report": args.stage11_run_report,
        "profile_path": args.profile,
        "manifest_path": args.manifest,
        "robustness_config": args.robustness_config,
        "augmentation_config": args.augmentation_config,
        "colab_runtime_version": args.colab_runtime_version,
        "stage7_evidence": args.stage7_evidence,
        "stage8_evidence": args.stage8_evidence,
        "stage10_evidence": args.stage10_evidence,
        "baseline_dir": args.baseline,
        "f4_bundle": args.f4_bundle,
        "f4_record": args.f4_record,
        "cache_a": args.cache_a,
        "cache_b": args.cache_b,
        "mfcc_reference": args.mfcc_reference}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "time-condition", "claim", "score"):
        command = commands.add_parser(name)
        command.add_argument("--model-kind", choices=MODEL_KINDS, required=True)
        command.add_argument("--registry", type=Path, required=True)
        command.add_argument("--clean-scores", type=Path, required=True)
        command.add_argument("--stage11-run-report", type=Path, required=True)
        command.add_argument("--profile", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--robustness-config", type=Path, default=Path(ROBUSTNESS_CONFIG))
        command.add_argument("--augmentation-config", type=Path,
                             default=Path("configs/augmentation.yaml"))
        command.add_argument("--colab-runtime-version", required=True)
        command.add_argument("--stage7-evidence", type=Path, required=True)
        command.add_argument("--stage8-evidence", type=Path, required=True)
        command.add_argument("--stage10-evidence", type=Path, required=True)
        command.add_argument("--baseline", type=Path)
        command.add_argument("--f4-bundle", type=Path)
        command.add_argument("--f4-record", type=Path)
        command.add_argument("--cache-a", type=Path)
        command.add_argument("--cache-b", type=Path)
        command.add_argument("--mfcc-reference", type=Path)
        if name in {"claim", "score"}:
            command.add_argument("--out", type=Path, required=True)
            command.add_argument("--approval-record", type=Path, required=True)
        if name == "claim":
            command.add_argument("--results-ref", required=True)
        elif name == "score":
            command.add_argument("--root", type=Path, required=True)
            command.add_argument("--run-id", required=True)
        elif name == "time-condition":
            command.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    inputs = _inputs(args)
    if args.command == "check":
        result = check(registry_path=args.registry, **inputs)
        print(_json_text(result), end="")
    elif args.command == "time-condition":
        result = time_validation_condition(root=args.root, registry_path=args.registry, **inputs)
        print(_json_text(result), end="")
    elif args.command == "claim":
        result = claim(
            registry_path=args.registry, out_dir=args.out, results_ref=args.results_ref,
            approval_record=args.approval_record, **inputs)
        print(_json_text(result), end="")
    else:
        result = score(
            root=args.root, registry_path=args.registry, out_dir=args.out,
            run_id=args.run_id, approval_record=args.approval_record, **inputs)
        print(_json_text(result), end="")


if __name__ == "__main__":
    main()
