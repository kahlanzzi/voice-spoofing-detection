"""Guarded Stage 13 preparation primitives (D35, D36, D48).

This module is the only code path permitted to reference the external dataset. It provides
synthetic-testable windowing, frozen-artifact preflight, and a dataset-wide one-shot claim.
It does not start a run by itself; the real run remains gated on F5 or the D48 cut-off and team
review. No dataset audio is opened by importing or testing this module.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from src.evaluation.bootstrap import BootstrapCI, bootstrap_ci
from src.evaluation.metrics import (
    BinaryMetrics,
    ThreeWayMetrics,
    binary_metrics,
    eer,
    roc_auc,
    three_way_metrics,
)
from src.evaluation.speaker_average import SpeakerMean, speaker_average
from src.evaluation.speakers import SpeakerMetrics, per_speaker_metrics
from src.freezing.manifests import safe_relative_path
from src.preprocessing import PreprocessingProfile, WaveformResult, preprocess_waveform
from src.risk.decisions import three_way_decisions

DATASET = "DEEP-VOICE"
CLAIM_SCHEMA = 1
D58_MFCC_REFERENCE_SHA256 = {
    "val-A": "04227c58b25f208771693db6262f39fd5276518811a13d60b9e7f9a560772bbd",
    "val-B": "04f91e3e4957ba25f1edd9b3b7af7b830249e4f617695b96d83674ac2d7783c7",
}
D58_MFCC_RTOL = 1e-5
D58_MFCC_ATOL = 1e-6
D48_CUTOFF = datetime(2026, 12, 4, 23, 59, tzinfo=ZoneInfo("Africa/Tunis"))
_SHA256 = re.compile(r"[0-9a-f]{64}")
SCORE_COLUMNS = (
    "file",
    "window_index",
    "source_speaker",
    "target_speaker",
    "label",
    "raw_score",
    "calibrated_p",
    "prediction",
    "kept_by_filter",
    "bundle_version",
    "run_id",
)
_PREDICTIONS = {"authentic", "synthetic", "uncertain"}


@dataclass(frozen=True)
class WindowScore:
    """One complete protocol window after the shared Stage 6 preprocessing call."""

    window_index: int
    raw_score: float
    speech_fraction: float
    padding_samples: int
    kept_by_filter: bool


@dataclass(frozen=True)
class WindowRun:
    """Scores and accounting from one in-memory recording pass."""

    scores: tuple[WindowScore, ...]
    n_dropped_samples: int

    @property
    def n_complete_windows(self) -> int:
        return len(self.scores)

    @property
    def filtered_scores(self) -> tuple[WindowScore, ...]:
        """The primary view: windows retained by the frozen speech filter."""
        return tuple(row for row in self.scores if row.kept_by_filter)

    @property
    def n_filter_dropped(self) -> int:
        return sum(not row.kept_by_filter for row in self.scores)


@dataclass(frozen=True)
class ExternalScoreRow:
    """One model's score for one external window; baseline-only fields are explicitly absent."""

    file: str
    window_index: int
    source_speaker: str
    target_speaker: str | None
    label: str
    raw_score: float
    calibrated_p: float | None
    prediction: str
    kept_by_filter: bool
    bundle_version: str | None
    run_id: str


@dataclass(frozen=True)
class ManifestFile:
    """Metadata-only expected file record used to validate complete score coverage."""

    file: str
    label: str
    source_speaker: str
    target_speaker: str | None
    duration_seconds: float
    sample_rate: int | None = None
    channels: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class FileDecision:
    """D34 decision for one file; no-window files keep undefined score and prediction."""

    file: str
    label: str
    source_speaker: str
    n_windows: int
    mean_p: float | None
    prediction: str | None


@dataclass(frozen=True)
class FileLevelView:
    """D34 file-level rows and summaries for one filtered or unfiltered view."""

    view: str
    rows: tuple[FileDecision, ...]
    counts_by_class: dict[str, dict[str, int]]
    counts_by_speaker: dict[str, dict[str, int]]
    metrics: dict[str, Any] | None


@dataclass(frozen=True)
class SpeakerLevelView:
    """D33/D19 eight-speaker table, macro means, undefined reasons and filter accounting."""

    view: str
    rows: tuple[SpeakerMetrics, ...]
    averages: dict[str, SpeakerMean]
    undefined_reasons: dict[str, dict[str, str]]
    filter_counts_by_class: dict[str, dict[str, int]]
    filter_counts_by_speaker: dict[str, dict[str, int]]


@dataclass(frozen=True)
class PairedEERView:
    """D36 paired CNN-minus-baseline EER result for one stored-score view."""

    view: str
    n_windows: int
    n_speakers: int
    cnn_eer: float | None
    baseline_eer: float | None
    difference: float | None
    interval: BootstrapCI | None
    n_resamples: int
    n_invalid: int
    undefined_reason: str | None


def _score_row_values(
    row: ExternalScoreRow, *, model_role: str, number: int
) -> tuple[str, ...]:
    try:
        canonical_file = safe_relative_path(row.file)
    except (TypeError, ValueError):
        raise ValueError(
            f"row {number}: file must be a safe relative POSIX path"
        ) from None
    if canonical_file != row.file:
        raise ValueError(
            f"row {number}: file path must be canonical POSIX: {canonical_file!r}"
        )
    if (
        isinstance(row.window_index, bool)
        or not isinstance(row.window_index, int)
        or row.window_index < 0
    ):
        raise ValueError(f"row {number}: window_index must be a non-negative integer")
    for name in ("source_speaker",):
        value = getattr(row, name)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"row {number}: {name} must be a non-empty trimmed string")
    if row.target_speaker is not None and (
        not isinstance(row.target_speaker, str)
        or not row.target_speaker.strip()
        or row.target_speaker != row.target_speaker.strip()
    ):
        raise ValueError(
            f"row {number}: target_speaker must be empty or a trimmed string"
        )
    if not isinstance(row.label, str) or row.label not in {"FAKE", "REAL"}:
        raise ValueError(f"row {number}: label must be FAKE or REAL")
    if (
        isinstance(row.raw_score, bool)
        or not isinstance(row.raw_score, Real)
        or not math.isfinite(row.raw_score)
    ):
        raise ValueError(f"row {number}: raw_score must be finite")
    if not isinstance(row.kept_by_filter, bool):
        raise TypeError(f"row {number}: kept_by_filter must be bool")
    if not isinstance(row.run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", row.run_id):
        raise ValueError(
            f"row {number}: run_id must be 32 lowercase hexadecimal characters"
        )
    if model_role == "baseline":
        if row.calibrated_p is not None or row.bundle_version is not None:
            raise ValueError(
                "baseline rows must not contain calibrated p or a bundle version"
            )
        if not isinstance(row.prediction, str) or row.prediction not in {
            "authentic",
            "synthetic",
        }:
            raise ValueError("baseline predictions must be forced binary (D29)")
        p_text = version_text = ""
    elif model_role == "cnn_bundle":
        if (
            isinstance(row.calibrated_p, bool)
            or not isinstance(row.calibrated_p, Real)
            or not math.isfinite(row.calibrated_p)
            or not 0 <= row.calibrated_p <= 1
        ):
            raise ValueError(f"row {number}: calibrated_p must be finite and in [0, 1]")
        if not isinstance(row.bundle_version, str) or not row.bundle_version.strip():
            raise ValueError("CNN rows need the verified bundle version")
        if not isinstance(row.prediction, str) or row.prediction not in _PREDICTIONS:
            raise ValueError(
                f"row {number}: invalid three-way prediction {row.prediction!r}"
            )
        p_text = repr(float(row.calibrated_p))
        version_text = row.bundle_version
    else:
        raise ValueError("model_role must be 'baseline' or 'cnn_bundle'")
    return (
        row.file,
        str(row.window_index),
        row.source_speaker,
        row.target_speaker or "",
        row.label,
        repr(float(row.raw_score)),
        p_text,
        row.prediction,
        "true" if row.kept_by_filter else "false",
        version_text,
        row.run_id,
    )


def write_external_score_file(
    out_dir: str | Path,
    *,
    model_id: str,
    model_role: str,
    model_artifact_sha256: str,
    manifest_sha256: str,
    rows: Iterator[ExternalScoreRow]
    | tuple[ExternalScoreRow, ...]
    | list[ExternalScoreRow],
) -> tuple[Path, Path]:
    """Write one deterministic per-window CSV and hash sidecar; refuse overwrite or mixed runs."""
    if not isinstance(model_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", model_id
    ):
        raise ValueError("model_id must be a safe filename token")
    if model_role not in {"baseline", "cnn_bundle"}:
        raise ValueError("model_role must be 'baseline' or 'cnn_bundle'")
    for field, value in (
        ("model_artifact_sha256", model_artifact_sha256),
        ("manifest_sha256", manifest_sha256),
    ):
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(f"{field} must be a lowercase SHA-256")
    try:
        rows = tuple(rows)
    except TypeError:
        raise ValueError("rows must be an iterable of ExternalScoreRow") from None
    if not rows or any(not isinstance(row, ExternalScoreRow) for row in rows):
        raise ValueError("rows must contain at least one ExternalScoreRow")
    values = [
        _score_row_values(row, model_role=model_role, number=n)
        for n, row in enumerate(rows, 1)
    ]
    identities = [(r.file, r.window_index) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate (file, window_index) score row")
    run_ids = {row.run_id for row in rows}
    if len(run_ids) != 1:
        raise ValueError("all score rows must belong to one run")
    versions = {row.bundle_version for row in rows}
    if len(versions) != 1:
        raise ValueError("all score rows must use one bundle version")
    values = sorted(values, key=lambda row: (row[0], int(row[1])))
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(SCORE_COLUMNS)
    writer.writerows(values)
    csv_bytes = buffer.getvalue().encode("utf-8")
    run_id = next(iter(run_ids))
    sidecar = {
        "schema_version": 1,
        "run_id": run_id,
        "model_id": model_id,
        "model_role": model_role,
        "model_artifact_sha256": model_artifact_sha256,
        "manifest_sha256": manifest_sha256,
        "bundle_version": next(iter(versions)),
        "columns": list(SCORE_COLUMNS),
        "n_rows": len(rows),
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
    }
    sidecar_bytes = (
        json.dumps(sidecar, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path, sidecar_path = (
        directory / f"{model_id}.csv",
        directory / f"{model_id}.json",
    )
    # Exclusive creation means a partial run cannot silently overwrite its evidence on retry.
    with csv_path.open("xb") as stream:
        stream.write(csv_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    with sidecar_path.open("xb") as stream:
        stream.write(sidecar_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    if csv_path.read_bytes() != csv_bytes or sidecar_path.read_bytes() != sidecar_bytes:
        raise OSError("stored score file read-back did not match written bytes")
    return csv_path, sidecar_path


def read_external_score_file(
    csv_path: str | Path,
    sidecar_path: str | Path | None = None,
) -> tuple[dict[str, Any], tuple[ExternalScoreRow, ...]]:
    """Read and verify a stored score file and its sidecar before downstream reporting."""
    csv_path = Path(csv_path)
    sidecar_path = (
        Path(sidecar_path)
        if sidecar_path is not None
        else csv_path.with_suffix(".json")
    )
    try:
        csv_bytes = csv_path.read_bytes()
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        csv_text = csv_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read stored score files: {exc}") from exc
    expected_sidecar = {
        "schema_version",
        "run_id",
        "model_id",
        "model_role",
        "model_artifact_sha256",
        "manifest_sha256",
        "bundle_version",
        "columns",
        "n_rows",
        "csv_sha256",
    }
    if not isinstance(sidecar, dict):
        raise TypeError("score sidecar must be an object")
    missing, unknown = (
        sorted(expected_sidecar - sidecar.keys()),
        sorted(sidecar.keys() - expected_sidecar),
    )
    if missing or unknown:
        raise ValueError(f"score sidecar keys: missing {missing}, unknown {unknown}")
    if sidecar["schema_version"] != 1 or isinstance(sidecar["schema_version"], bool):
        raise ValueError("unsupported score sidecar schema")
    if sidecar["columns"] != list(SCORE_COLUMNS):
        raise ValueError(
            "score sidecar columns do not match the frozen Stage 13 schema"
        )
    if (
        not isinstance(sidecar["n_rows"], int)
        or isinstance(sidecar["n_rows"], bool)
        or sidecar["n_rows"] < 1
    ):
        raise ValueError("score sidecar n_rows must be a positive integer")
    if hashlib.sha256(csv_bytes).hexdigest() != sidecar["csv_sha256"]:
        raise ValueError("stored score CSV SHA-256 does not match its sidecar")
    role = sidecar["model_role"]
    if not isinstance(role, str) or role not in {"baseline", "cnn_bundle"}:
        raise ValueError("invalid score sidecar model_role")
    if not isinstance(sidecar["model_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", sidecar["model_id"]
    ):
        raise ValueError("invalid score sidecar model_id")
    for field in ("run_id",):
        if not isinstance(sidecar[field], str) or not re.fullmatch(
            r"[0-9a-f]{32}", sidecar[field]
        ):
            raise ValueError(f"invalid score sidecar {field}")
    for field in ("model_artifact_sha256", "manifest_sha256"):
        if not isinstance(sidecar[field], str) or not _SHA256.fullmatch(sidecar[field]):
            raise ValueError(f"invalid score sidecar {field}")
    try:
        parsed = list(csv.reader(io.StringIO(csv_text, newline="")))
    except csv.Error as exc:
        raise ValueError(f"invalid stored score CSV: {exc}") from None
    if not parsed or tuple(parsed[0]) != SCORE_COLUMNS:
        raise ValueError(
            "stored score CSV header does not match the frozen Stage 13 schema"
        )
    if len(parsed) - 1 != sidecar["n_rows"]:
        raise ValueError("stored score CSV row count does not match its sidecar")
    rows = []
    for number, cells in enumerate(parsed[1:], 1):
        if len(cells) != len(SCORE_COLUMNS):
            raise ValueError(
                f"row {number}: expected {len(SCORE_COLUMNS)} columns, got {len(cells)}"
            )
        row = dict(zip(SCORE_COLUMNS, cells))
        if row["kept_by_filter"] not in {"true", "false"}:
            raise ValueError(f"row {number}: kept_by_filter must be true or false")
        try:
            score = ExternalScoreRow(
                file=row["file"],
                window_index=int(row["window_index"]),
                source_speaker=row["source_speaker"],
                target_speaker=row["target_speaker"] or None,
                label=row["label"],
                raw_score=float(row["raw_score"]),
                calibrated_p=None
                if row["calibrated_p"] == ""
                else float(row["calibrated_p"]),
                prediction=row["prediction"],
                kept_by_filter=row["kept_by_filter"] == "true",
                bundle_version=row["bundle_version"] or None,
                run_id=row["run_id"],
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid stored score row {number}: {exc}") from None
        canonical = _score_row_values(score, model_role=role, number=number)
        if tuple(cells) != canonical:
            raise ValueError(f"row {number}: stored values are not canonical")
        if score.run_id != sidecar["run_id"]:
            raise ValueError(f"row {number}: run ID differs from the sidecar")
        rows.append(score)
    if role == "baseline" and sidecar["bundle_version"] is not None:
        raise ValueError("baseline score sidecar must not claim a bundle version")
    if role == "cnn_bundle" and not isinstance(sidecar["bundle_version"], str):
        raise ValueError("CNN score sidecar needs a bundle version")
    if any(row.bundle_version != sidecar["bundle_version"] for row in rows):
        raise ValueError("score row bundle version differs from the sidecar")
    identities = [(row.file, row.window_index) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate (file, window_index) score row")
    return sidecar, tuple(rows)


def rows_from_window_run(
    window_run: WindowRun,
    *,
    file: str,
    source_speaker: str,
    target_speaker: str | None,
    label: str,
    run_id: str,
    model_role: str,
    bundle_version: str | None = None,
    policy: Any | None = None,
    baseline_threshold: float | None = None,
) -> tuple[ExternalScoreRow, ...]:
    """Apply frozen CNN policy or D29 baseline threshold to one model's shared-window scores."""
    if not isinstance(window_run, WindowRun):
        raise TypeError("window_run must be a WindowRun")
    raw = np.asarray([score.raw_score for score in window_run.scores], dtype=float)
    if model_role == "cnn_bundle":
        if policy is None or baseline_threshold is not None:
            raise ValueError(
                "CNN rows need the bundle risk policy and no baseline threshold"
            )
        decisions = tuple(policy.decide(raw))
        if len(decisions) != len(raw):
            raise ValueError(
                "risk policy returned a different number of decisions than windows"
            )
        probabilities = [decision.p for decision in decisions]
        predictions = [decision.prediction for decision in decisions]
        if bundle_version is None:
            raise ValueError("CNN rows need the verified bundle version")
    elif model_role == "baseline":
        if (
            policy is not None
            or baseline_threshold is None
            or bundle_version is not None
        ):
            raise ValueError("baseline rows need only their frozen D29 threshold")
        predictions = three_way_decisions(raw, baseline_threshold, 0.0).tolist()
        probabilities = [None] * len(raw)
    else:
        raise ValueError("model_role must be 'baseline' or 'cnn_bundle'")
    return tuple(
        ExternalScoreRow(
            file=file,
            window_index=score.window_index,
            source_speaker=source_speaker,
            target_speaker=target_speaker,
            label=label,
            raw_score=score.raw_score,
            calibrated_p=probability,
            prediction=str(prediction),
            kept_by_filter=score.kept_by_filter,
            bundle_version=bundle_version,
            run_id=run_id,
        )
        for score, probability, prediction in zip(
            window_run.scores, probabilities, predictions
        )
    )


def _validate_complete_external_rows(rows, manifest):
    """Validate stored window identities and return metadata groups plus inferred model role."""
    score_rows, manifest_rows = tuple(rows), tuple(manifest)
    if not score_rows or any(
        not isinstance(row, ExternalScoreRow) for row in score_rows
    ):
        raise ValueError("score rows must contain ExternalScoreRow entries")
    if not manifest_rows or any(
        not isinstance(item, ManifestFile) for item in manifest_rows
    ):
        raise ValueError("manifest must contain ManifestFile entries")
    first = score_rows[0]
    role = (
        "baseline"
        if first.calibrated_p is None and first.bundle_version is None
        else "cnn_bundle"
    )
    manifest_by_file: dict[str, tuple[ManifestFile, int]] = {}
    for item in manifest_rows:
        try:
            canonical = safe_relative_path(item.file)
        except (TypeError, ValueError):
            raise ValueError(
                "manifest file must be a safe relative POSIX path"
            ) from None
        if canonical != item.file:
            raise ValueError(
                f"manifest file path must be canonical POSIX: {canonical!r}"
            )
        if item.file in manifest_by_file:
            raise ValueError(f"duplicate manifest file {item.file!r}")
        if (
            not isinstance(item.label, str)
            or item.label not in {"FAKE", "REAL"}
            or not isinstance(item.source_speaker, str)
            or not item.source_speaker.strip()
            or item.source_speaker != item.source_speaker.strip()
        ):
            raise ValueError(f"invalid label or source speaker for {item.file!r}")
        if item.target_speaker is not None and (
            not isinstance(item.target_speaker, str)
            or not item.target_speaker.strip()
            or item.target_speaker != item.target_speaker.strip()
        ):
            raise ValueError(f"invalid target speaker for {item.file!r}")
        if (
            not isinstance(item.duration_seconds, Real)
            or isinstance(item.duration_seconds, bool)
            or not math.isfinite(item.duration_seconds)
            or item.duration_seconds < 0
        ):
            raise ValueError(f"invalid duration for {item.file!r}")
        manifest_by_file[item.file] = (item, math.floor(item.duration_seconds / 2.0))
    grouped: dict[str, list[ExternalScoreRow]] = {key: [] for key in manifest_by_file}
    for row in score_rows:
        if not isinstance(row.file, str):
            raise TypeError("score row file must be a string")
        if isinstance(row.window_index, bool) or not isinstance(row.window_index, int):
            raise TypeError("score row window_index must be an integer")
    identities = [(row.file, row.window_index) for row in score_rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate (file, window_index) score row")
    run_ids, versions = set(), set()
    for number, row in enumerate(score_rows, 1):
        if row.file not in manifest_by_file:
            raise ValueError(f"score row references unexpected file {row.file!r}")
        item, _ = manifest_by_file[row.file]
        if (row.label, row.source_speaker, row.target_speaker) != (
            item.label,
            item.source_speaker,
            item.target_speaker,
        ):
            raise ValueError(f"score metadata differs from manifest for {row.file!r}")
        _score_row_values(row, model_role=role, number=number)
        run_ids.add(row.run_id)
        versions.add(row.bundle_version)
        grouped[row.file].append(row)
    if len(run_ids) != 1 or len(versions) != 1:
        raise ValueError("score rows mix run IDs or model versions")
    for file, (_, expected) in manifest_by_file.items():
        score_group = sorted(grouped[file], key=lambda row: row.window_index)
        if len(score_group) != expected or [
            row.window_index for row in score_group
        ] != list(range(expected)):
            raise ValueError(f"incomplete or unexpected window indices for {file!r}")
    return score_rows, manifest_by_file, grouped, role


def file_level_view(
    rows: Iterator[ExternalScoreRow]
    | tuple[ExternalScoreRow, ...]
    | list[ExternalScoreRow],
    manifest: Iterator[ManifestFile] | tuple[ManifestFile, ...] | list[ManifestFile],
    *,
    threshold: float,
    half_width: float,
    view: str,
    n_resamples: int = 2000,
    seed: int = 2026,
) -> FileLevelView:
    """Compute D34's CNN file means from stored rows only; never opens audio or re-scores.

    Input rows must contain every complete window listed by manifest duration. Filtered/unfiltered
    views select from the same rows. Empty selections remain explicit no-window files.
    """
    if view not in {"filtered", "unfiltered"}:
        raise ValueError("view must be 'filtered' or 'unfiltered'")
    if (
        not isinstance(threshold, Real)
        or isinstance(threshold, bool)
        or not math.isfinite(threshold)
    ):
        raise ValueError("threshold must be finite")
    if (
        not isinstance(half_width, Real)
        or isinstance(half_width, bool)
        or not math.isfinite(half_width)
        or half_width < 0
    ):
        raise ValueError("half_width must be finite and >= 0")
    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, int)
        or n_resamples < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError("n_resamples must be positive and seed must be an integer")
    _score_rows, manifest_by_file, grouped, role = _validate_complete_external_rows(
        rows, manifest
    )
    if role != "cnn_bundle":
        raise ValueError("D34 file-level decisions apply to the frozen CNN bundle only")

    output = []
    counts_by_class = {
        label: {"expected": 0, "evaluable": 0, "no_window": 0}
        for label in ("REAL", "FAKE")
    }
    counts_by_speaker: dict[str, dict[str, int]] = {}
    eval_labels, eval_ps, eval_predictions = [], [], []
    for file in sorted(manifest_by_file):
        item, expected = manifest_by_file[file]
        scores = sorted(grouped[file], key=lambda row: row.window_index)
        if len(scores) != expected or [row.window_index for row in scores] != list(
            range(expected)
        ):
            raise ValueError(f"incomplete or unexpected window indices for {file!r}")
        selected = (
            scores
            if view == "unfiltered"
            else [row for row in scores if row.kept_by_filter]
        )
        counts = counts_by_class[item.label]
        counts["expected"] += 1
        speaker_counts = counts_by_speaker.setdefault(
            item.source_speaker, {"expected": 0, "evaluable": 0, "no_window": 0}
        )
        speaker_counts["expected"] += 1
        if not selected:
            counts["no_window"] += 1
            speaker_counts["no_window"] += 1
            output.append(
                FileDecision(file, item.label, item.source_speaker, 0, None, None)
            )
            continue
        mean_p = float(
            np.mean([row.calibrated_p for row in selected], dtype=np.float64)
        )
        prediction = str(three_way_decisions([mean_p], threshold, half_width)[0])
        counts["evaluable"] += 1
        speaker_counts["evaluable"] += 1
        output.append(
            FileDecision(
                file, item.label, item.source_speaker, len(selected), mean_p, prediction
            )
        )
        eval_labels.append(1 if item.label == "FAKE" else 0)
        eval_ps.append(mean_p)
        eval_predictions.append(prediction)

    if not eval_labels:
        metrics = None
    else:
        labels = np.asarray(eval_labels, dtype=int)
        scores = np.asarray(eval_ps, dtype=float)
        both = len(np.unique(labels)) == 2
        binary = binary_metrics(labels, scores, threshold)
        threeway = three_way_metrics(labels, eval_predictions)
        metrics = {
            "n_files": len(eval_labels),
            "n_speakers": len({row.source_speaker for row in output if row.n_windows}),
            "eer": eer(labels, scores)[0] if both else None,
            "roc_auc": roc_auc(labels, scores) if both else None,
            "binary": binary,
            "three_way": threeway,
        }
        clusters = [row.source_speaker for row in output if row.n_windows]
        decisions = np.asarray(eval_predictions, dtype=object)
        interval_names = ["eer", "roc_auc"]
        interval_names.extend(
            f"binary.{metric}"
            for metric in ("accuracy", "precision", "recall", "f1", "fpr", "fnr")
        )
        interval_names.extend(
            f"three_way.{metric}"
            for metric in (
                "coverage",
                "deferral_rate",
                "deferral_rate_bonafide",
                "deferral_rate_spoof",
            )
        )
        interval_names.extend(
            f"three_way.non_deferred.{metric}"
            for metric in (
                "accuracy",
                "precision",
                "recall",
                "f1",
                "fpr",
                "fnr",
            )
        )
        intervals = {}
        for name in interval_names:

            def statistic(indices: np.ndarray, metric_name=name) -> float:
                selected_y, selected_scores = labels[indices], scores[indices]
                if metric_name == "eer":
                    return eer(selected_y, selected_scores)[0]
                if metric_name == "roc_auc":
                    return roc_auc(selected_y, selected_scores)
                if metric_name.startswith("binary."):
                    result = binary_metrics(selected_y, selected_scores, threshold)
                    return float(getattr(result, metric_name.split(".", 1)[1]))
                result = three_way_metrics(selected_y, decisions[indices])
                if metric_name.startswith("three_way.non_deferred."):
                    result = result.non_deferred
                    field_name = metric_name.rsplit(".", 1)[1]
                else:
                    field_name = metric_name.split(".", 1)[1]
                return float(getattr(result, field_name))

            try:
                estimate = float(statistic(np.arange(len(labels))))
            except ValueError as exc:
                intervals[name] = {
                    "estimate": None,
                    "low": None,
                    "high": None,
                    "n_resamples": n_resamples,
                    "n_invalid": 0,
                    "undefined_reason": str(exc),
                }
                continue
            if not math.isfinite(estimate):
                intervals[name] = {
                    "estimate": None,
                    "low": None,
                    "high": None,
                    "n_resamples": n_resamples,
                    "n_invalid": 0,
                    "undefined_reason": "metric is undefined on the full evaluable file set",
                }
                continue
            try:
                ci = bootstrap_ci(
                    statistic,
                    len(labels),
                    clusters=clusters,
                    n_resamples=n_resamples,
                    seed=seed,
                )
            except ValueError as exc:
                if "all " not in str(exc) or "resamples are invalid" not in str(exc):
                    raise
                intervals[name] = {
                    "estimate": estimate,
                    "low": None,
                    "high": None,
                    "n_resamples": n_resamples,
                    "n_invalid": n_resamples,
                    "undefined_reason": "all speaker-clustered resamples had an undefined metric",
                }
                continue
            intervals[name] = {
                "estimate": estimate,
                "low": ci.low,
                "high": ci.high,
                "n_resamples": ci.n_resamples,
                "n_invalid": ci.n_invalid,
                "undefined_reason": None,
            }
        metrics["intervals"] = intervals
    return FileLevelView(
        view, tuple(output), counts_by_class, counts_by_speaker, metrics
    )


def _empty_binary_metrics() -> BinaryMetrics:
    nan = float("nan")
    return BinaryMetrics(0, 0, 0, 0, 0, 0, nan, nan, nan, nan, nan, nan)


def _empty_speaker_metrics(speaker: str) -> SpeakerMetrics:
    nan = float("nan")
    binary = _empty_binary_metrics()
    three_way = ThreeWayMetrics(
        confusion=((0, 0, 0), (0, 0, 0)),
        n=0,
        n_deferred=0,
        coverage=nan,
        deferral_rate=nan,
        deferral_rate_bonafide=nan,
        deferral_rate_spoof=nan,
        non_deferred=binary,
    )
    return SpeakerMetrics(speaker, 0, nan, nan, binary, three_way)


def speaker_level_view(
    rows: Iterator[ExternalScoreRow]
    | tuple[ExternalScoreRow, ...]
    | list[ExternalScoreRow],
    manifest: Iterator[ManifestFile] | tuple[ManifestFile, ...] | list[ManifestFile],
    *,
    model_role: str,
    threshold: float,
    half_width: float,
    view: str,
) -> SpeakerLevelView:
    """Build D19 rows for every manifest speaker and D33 propagated equal-speaker averages."""
    if view not in {"filtered", "unfiltered"}:
        raise ValueError("view must be 'filtered' or 'unfiltered'")
    if model_role not in {"baseline", "cnn_bundle"}:
        raise ValueError("model_role must be 'baseline' or 'cnn_bundle'")
    if model_role == "baseline" and half_width != 0:
        raise ValueError("baseline-v1 has no uncertain band; half_width must be zero")
    if (
        not isinstance(threshold, Real)
        or isinstance(threshold, bool)
        or not math.isfinite(threshold)
    ):
        raise ValueError("threshold must be finite")
    if (
        not isinstance(half_width, Real)
        or isinstance(half_width, bool)
        or not math.isfinite(half_width)
        or half_width < 0
    ):
        raise ValueError("half_width must be finite and >= 0")
    score_rows, manifest_by_file, _grouped, inferred_role = (
        _validate_complete_external_rows(rows, manifest)
    )
    if model_role != inferred_role:
        raise ValueError("requested model_role differs from stored score rows")
    roster = sorted({item.source_speaker for item, _ in manifest_by_file.values()})
    if len(roster) != 8:
        raise ValueError(
            f"D19 expects exactly eight manifest source speakers, found {len(roster)}"
        )

    selected_rows = [
        row for row in score_rows if view == "unfiltered" or row.kept_by_filter
    ]
    score_values = [
        row.calibrated_p if model_role == "cnn_bundle" else row.raw_score
        for row in selected_rows
    ]
    labels = [1 if row.label == "FAKE" else 0 for row in selected_rows]
    speakers = [row.source_speaker for row in selected_rows]
    if selected_rows:
        calculated = per_speaker_metrics(
            labels, score_values, speakers, threshold=threshold, half_width=half_width
        )
        rows_by_speaker = {row.source_speaker: row for row in calculated}
    else:
        rows_by_speaker = {}
    speaker_rows = tuple(
        rows_by_speaker.get(speaker, _empty_speaker_metrics(speaker))
        for speaker in roster
    )
    averages = speaker_average(speaker_rows, undefined_policy="propagate")

    undefined_reasons = {}
    for metric, summary in averages.items():
        if not summary.undefined_speakers:
            continue
        reasons = {}
        for speaker in summary.undefined_speakers:
            speaker_row = rows_by_speaker.get(speaker)
            if speaker_row is None or speaker_row.n == 0:
                reason = "no windows retained in this view"
            elif metric in {"eer", "roc_auc"}:
                reason = "only one label class is present for this speaker"
            else:
                reason = "the rate denominator is zero for this speaker"
            reasons[speaker] = reason
        undefined_reasons[metric] = reasons

    def counts_template():
        return {"complete_windows": 0, "kept": 0, "dropped": 0, "retention_rate": None}

    by_class = {"REAL": counts_template(), "FAKE": counts_template()}
    by_speaker = {speaker: counts_template() for speaker in roster}
    for row in score_rows:
        for bucket in (by_class[row.label], by_speaker[row.source_speaker]):
            bucket["complete_windows"] += 1
            bucket["kept" if row.kept_by_filter else "dropped"] += 1
    for bucket in (*by_class.values(), *by_speaker.values()):
        total = bucket["complete_windows"]
        bucket["retention_rate"] = bucket["kept"] / total if total else None
    return SpeakerLevelView(
        view, speaker_rows, averages, undefined_reasons, by_class, by_speaker
    )


def paired_eer_comparison(
    cnn_score_file: tuple[
        Mapping[str, Any], Iterator[ExternalScoreRow] | tuple[ExternalScoreRow, ...]
    ],
    baseline_score_file: tuple[
        Mapping[str, Any], Iterator[ExternalScoreRow] | tuple[ExternalScoreRow, ...]
    ],
    manifest: Iterator[ManifestFile] | tuple[ManifestFile, ...] | list[ManifestFile],
    *,
    view: str,
    n_resamples: int = 2000,
    seed: int = 2026,
    second_model_role: str = "baseline",
) -> PairedEERView:
    """Compute a paired EER difference from verified stored rows; never opens audio.

    The two inputs are the `(sidecar, rows)` tuples returned by `read_external_score_file`.
    Both scores use one set of row indices in every source-speaker bootstrap resample. The
    sidecars bind the comparison to the same manifest and dataset-wide run claim. The default
    role pair implements D36 (CNN minus baseline); H8 passes `cnn_bundle` for the second role.
    """
    if view not in {"filtered", "unfiltered"}:
        raise ValueError("view must be 'filtered' or 'unfiltered'")
    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, int)
        or n_resamples < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError("n_resamples must be positive and seed must be an integer")
    try:
        cnn_meta, cnn_rows = cnn_score_file
        baseline_meta, baseline_rows = baseline_score_file
    except (TypeError, ValueError):
        raise ValueError("score files must be (sidecar, rows) pairs") from None
    if not isinstance(cnn_meta, Mapping) or not isinstance(baseline_meta, Mapping):
        raise TypeError("score sidecars must be mappings")
    if second_model_role not in {"baseline", "cnn_bundle"}:
        raise ValueError("second_model_role must be 'baseline' or 'cnn_bundle'")
    for meta, role in ((cnn_meta, "cnn_bundle"), (baseline_meta, second_model_role)):
        if meta.get("model_role") != role:
            raise ValueError(f"expected a {role} stored-score file")
    for key in ("run_id", "manifest_sha256"):
        if cnn_meta.get(key) != baseline_meta.get(key):
            raise ValueError(f"CNN and baseline score files do not share {key}")

    _cnn, _cnn_manifest, _cnn_grouped, cnn_role = _validate_complete_external_rows(
        cnn_rows, manifest
    )
    _baseline, _baseline_manifest, _baseline_grouped, baseline_role = (
        _validate_complete_external_rows(baseline_rows, manifest)
    )
    if cnn_role != "cnn_bundle" or baseline_role != second_model_role:
        raise ValueError("stored score row roles do not match their sidecars")
    cnn_by_key = {(row.file, row.window_index): row for row in cnn_rows}
    baseline_by_key = {(row.file, row.window_index): row for row in baseline_rows}
    if cnn_by_key.keys() != baseline_by_key.keys():
        raise ValueError("CNN and baseline stored rows do not cover the same windows")
    paired = []
    for key in sorted(cnn_by_key):
        cnn, baseline = cnn_by_key[key], baseline_by_key[key]
        if (
            cnn.run_id,
            cnn.label,
            cnn.source_speaker,
            cnn.target_speaker,
            cnn.kept_by_filter,
        ) != (
            baseline.run_id,
            baseline.label,
            baseline.source_speaker,
            baseline.target_speaker,
            baseline.kept_by_filter,
        ):
            raise ValueError(f"CNN and baseline row metadata differs for {key}")
        if cnn.run_id != cnn_meta["run_id"]:
            raise ValueError(f"score row run ID differs from its sidecar for {key}")
        if view == "unfiltered" or cnn.kept_by_filter:
            paired.append((cnn, baseline))

    n_windows = len(paired)
    n_speakers = len({cnn.source_speaker for cnn, _ in paired})
    if not paired:
        return PairedEERView(
            view,
            0,
            0,
            None,
            None,
            None,
            None,
            n_resamples,
            0,
            "no windows retained in this view",
        )
    labels = np.asarray(
        [1 if cnn.label == "FAKE" else 0 for cnn, _ in paired], dtype=int
    )
    cnn_scores = np.asarray([cnn.raw_score for cnn, _ in paired], dtype=float)
    baseline_scores = np.asarray(
        [baseline.raw_score for _, baseline in paired], dtype=float
    )
    clusters = [cnn.source_speaker for cnn, _ in paired]
    if len(np.unique(labels)) < 2:
        return PairedEERView(
            view,
            n_windows,
            n_speakers,
            None,
            None,
            None,
            None,
            n_resamples,
            0,
            "selected windows contain only one label class",
        )
    cnn_eer = eer(labels, cnn_scores)[0]
    baseline_eer = eer(labels, baseline_scores)[0]
    difference = cnn_eer - baseline_eer

    def paired_difference(indices: np.ndarray) -> float:
        return (
            eer(labels[indices], cnn_scores[indices])[0]
            - eer(labels[indices], baseline_scores[indices])[0]
        )

    try:
        interval = bootstrap_ci(
            paired_difference,
            n_windows,
            clusters=clusters,
            n_resamples=n_resamples,
            seed=seed,
        )
    except ValueError as exc:
        if "all " not in str(exc) or "resamples are invalid" not in str(exc):
            raise
        return PairedEERView(
            view,
            n_windows,
            n_speakers,
            cnn_eer,
            baseline_eer,
            difference,
            None,
            n_resamples,
            n_resamples,
            "all speaker-clustered resamples lacked both label classes",
        )
    return PairedEERView(
        view,
        n_windows,
        n_speakers,
        cnn_eer,
        baseline_eer,
        difference,
        interval,
        interval.n_resamples,
        interval.n_invalid,
        None,
    )


def strict_json_dumps(value: Any) -> str:
    """Serialize Stage 13 report data as strict JSON, mapping undefined numeric values to null."""

    def clean(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, np.ndarray):
            return clean(item.tolist())
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, datetime):
            return item.isoformat()
        if is_dataclass(item) and not isinstance(item, type):
            return {
                field.name: clean(getattr(item, field.name)) for field in fields(item)
            }
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("strict JSON report mappings must use string keys")
            return {key: clean(nested) for key, nested in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(nested) for nested in item]
        raise TypeError(f"unsupported strict JSON report value: {type(item).__name__}")

    return json.dumps(clean(value), sort_keys=True, indent=2, allow_nan=False) + "\n"


def write_external_report(
    path: str | Path, report: Mapping[str, Any]
) -> tuple[Path, str]:
    """Write a strict, immutable JSON report and verify the exact bytes by read-back."""
    target = Path(path)
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    data = strict_json_dumps(report).encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    verified = target.read_bytes()
    if verified != data:
        raise OSError("report read-back did not match the bytes written")
    return target, hashlib.sha256(verified).hexdigest()


@dataclass(frozen=True)
class FrozenArtifacts:
    """Loaded and anchored F4 artifacts; profile and policy come from verified bundle bytes."""

    bundle_model: Any
    baseline_model: Any
    bundle_manifest_sha256: str
    baseline_artifact_sha256: str
    v2_model: Any | None = None
    v2_manifest_sha256: str | None = None


@dataclass(frozen=True)
class PreparedRun:
    """Verified artifacts plus the already-persisted claim required before scoring."""

    artifacts: FrozenArtifacts
    manifest_files: tuple[ManifestFile, ...]
    manifest_sha256: str
    claim_path: Path
    claim_sha256: str
    d58_report_sha256: str
    scorers: Mapping[str, Callable[[np.ndarray], Real]]
    run_id: str
    model_set: tuple[str, ...] = ("bundle-v1", "baseline-v1")


@dataclass(frozen=True)
class GuardedRunResult:
    """Verified products and claim evidence from one completed guarded pass."""

    score_files: dict[str, tuple[Path, Path]]
    report_path: Path
    report_sha256: str
    claim_path: Path
    claim_sha256: str
    claim_line_count: int
    files_scored: int
    complete_windows: int
    dropped_samples: int
    filter_dropped_windows: int


def _sample_axis(waveform: np.ndarray, channel_axis: int) -> int:
    if waveform.ndim == 1:
        return 0
    if waveform.ndim != 2:
        raise ValueError(
            f"waveform must be one- or two-dimensional, got shape {waveform.shape}"
        )
    axis = channel_axis if channel_axis >= 0 else waveform.ndim + channel_axis
    if axis not in (0, 1):
        raise ValueError(f"channel_axis must identify axis 0 or 1, got {channel_axis}")
    return 1 - axis


def _window_count(
    sample_count: int, sample_rate: int, profile: PreprocessingProfile
) -> tuple[int, int]:
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, np.integer))
        or sample_rate <= 0
    ):
        raise ValueError(f"sample_rate must be a positive integer, got {sample_rate!r}")
    duration = float(profile.waveform["duration_seconds"])
    window_samples = round(duration * int(sample_rate))
    if window_samples <= 0 or not np.isclose(
        window_samples, duration * int(sample_rate)
    ):
        raise ValueError(
            "profile duration does not map to a whole number of input samples"
        )
    return sample_count // window_samples, window_samples


def iter_complete_windows(
    waveform: np.ndarray,
    sample_rate: int,
    profile: PreprocessingProfile,
    *,
    channel_axis: int = -1,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield complete, non-overlapping windows from file start; drop a short remainder."""
    array = np.asarray(waveform)
    sample_axis = _sample_axis(array, channel_axis)
    count, window_samples = _window_count(
        array.shape[sample_axis], sample_rate, profile
    )
    for index in range(count):
        start = index * window_samples
        slices = [slice(None)] * array.ndim
        slices[sample_axis] = slice(start, start + window_samples)
        yield index, array[tuple(slices)]


def score_windows(
    waveform: np.ndarray,
    sample_rate: int,
    profile: PreprocessingProfile,
    scorer: Callable[[np.ndarray], Real],
    *,
    channel_axis: int = -1,
) -> WindowRun:
    """Single-model convenience wrapper; prefer score_window_models for the joint run."""
    return score_window_models(
        waveform, sample_rate, profile, {"model": scorer}, channel_axis=channel_axis
    )["model"]


def score_window_models(
    waveform: np.ndarray,
    sample_rate: int,
    profile: PreprocessingProfile,
    scorers: Mapping[str, Callable[[np.ndarray], Real]],
    *,
    channel_axis: int = -1,
) -> dict[str, WindowRun]:
    """Preprocess each complete window once, then call every model on that same prepared data."""
    if not isinstance(scorers, Mapping) or not scorers:
        raise ValueError("scorers must be a non-empty mapping")
    if any(
        not isinstance(name, str) or not name.strip() or not callable(fn)
        for name, fn in scorers.items()
    ):
        raise TypeError("scorers must map non-empty model names to callables")
    array = np.asarray(waveform)
    sample_axis = _sample_axis(array, channel_axis)
    count, window_samples = _window_count(
        array.shape[sample_axis], sample_rate, profile
    )
    rows = {name: [] for name in scorers}
    for index, raw_window in iter_complete_windows(
        array, sample_rate, profile, channel_axis=channel_axis
    ):
        prepared: WaveformResult = preprocess_waveform(
            raw_window, sample_rate, profile, channel_axis=channel_axis
        )
        for name, scorer in scorers.items():
            value = scorer(prepared.waveform)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise ValueError(
                    f"scorer {name!r} must return one finite numeric score, got {value!r}"
                )
            rows[name].append(
                WindowScore(
                    window_index=index,
                    raw_score=float(value),
                    speech_fraction=float(prepared.speech_fraction),
                    padding_samples=int(prepared.padding_samples),
                    kept_by_filter=(
                        prepared.speech_fraction
                        >= float(profile.silence["external_min_speech_fraction"])
                    ),
                )
            )
    dropped = array.shape[sample_axis] - count * window_samples
    return {name: WindowRun(tuple(scores), dropped) for name, scores in rows.items()}


def load_external_config(path: str | Path) -> dict[str, Any]:
    """Load a strict template; all paths are supplied by the reviewed invocation."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read external-run config {path}: {exc}") from exc
    expected = {
        "schema_version",
        "dataset",
        "registry_dir",
        "bundle_dir",
        "baseline_dir",
        "f4_record",
        "manifest_path",
        "manifest_sha256",
        "output_dir",
        "d58_rehearsal_report",
        "d58_rehearsal_report_sha256",
    }
    optional = {"v2_bundle_dir", "f5_record"}
    if not isinstance(raw, dict):
        raise TypeError("external-run config must be a mapping")
    missing, unknown = (
        sorted(expected - raw.keys()),
        sorted(raw.keys() - expected - optional),
    )
    if missing or unknown:
        raise ValueError(
            f"external-run config keys: missing {missing}, unknown {unknown}"
        )
    if raw["schema_version"] != 1 or isinstance(raw["schema_version"], bool):
        raise ValueError("schema_version must be 1")
    if raw["dataset"] != DATASET:
        raise ValueError(f"dataset must be {DATASET!r}")
    if not isinstance(raw["manifest_sha256"], str) or not _SHA256.fullmatch(
        raw["manifest_sha256"]
    ):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256")
    for key in (expected | (optional & raw.keys())) - {
        "schema_version",
        "dataset",
        "manifest_sha256",
    }:
        if key == "d58_rehearsal_report_sha256":
            if not isinstance(raw[key], str) or not _SHA256.fullmatch(raw[key]):
                raise ValueError(
                    "d58_rehearsal_report_sha256 must be lowercase SHA-256"
                )
            continue
        if key in optional and raw[key] is None:
            continue
        if (
            not isinstance(raw[key], str)
            or not raw[key].strip()
            or raw[key].startswith("<")
        ):
            raise ValueError(f"{key} must be a concrete non-empty path")
        raw[key] = Path(raw[key])
    return raw


def verify_f4_artifacts(
    bundle_dir: str | Path,
    baseline_dir: str | Path,
    f4_record: str | Path,
    *,
    bundle_loader: Callable[[Path], Any] | None = None,
    bundle_verifier: Callable[[Path], Any] | None = None,
    baseline_loader: Callable[[Path], Any] | None = None,
    baseline_hash: Callable[[Path], str] | None = None,
) -> FrozenArtifacts:
    """Verify bundle and baseline against F4 before callers may claim or decode any audio."""
    try:
        record = json.loads(Path(f4_record).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read F4 record: {exc}") from exc
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("checks"), list)
        or not record["checks"]
    ):
        raise TypeError("F4 record is missing its checks")
    if not all(
        isinstance(item, dict) and item.get("ok") is True for item in record["checks"]
    ):
        raise ValueError("F4 record contains a failed or malformed check")

    if bundle_verifier is None:
        from src.bundle.bundle import load_bundle

        bundle_verifier = load_bundle
    if bundle_loader is None:
        from src.bundle.model_loader import load_bundle_model

        bundle_loader = load_bundle_model
    if baseline_loader is None:
        from src.models.baseline import load

        baseline_loader = load
    if baseline_hash is None:
        from src.models.baseline import artifact_sha256

        baseline_hash = artifact_sha256

    # Verify the bundle and its F4 anchor before loading checkpoint bytes into a model.
    bundle = bundle_verifier(Path(bundle_dir))
    expected_bundle_sha = record.get("manifest_sha256")
    if bundle.manifest_sha256 != expected_bundle_sha:
        raise ValueError("ModelBundle manifest does not match the F4 record")
    if bundle.manifest.bundle_version != record.get("bundle_version"):
        raise ValueError("ModelBundle version does not match the F4 record")

    expected_baseline = record.get("baseline")
    if not isinstance(expected_baseline, dict):
        raise TypeError("F4 record has no baseline anchor")
    actual_baseline_sha = baseline_hash(Path(baseline_dir))
    if actual_baseline_sha != expected_baseline.get("artifact_sha256"):
        raise ValueError("baseline artifact does not match the F4 record")
    baseline_model = baseline_loader(Path(baseline_dir))
    if baseline_model.version != expected_baseline.get("version"):
        raise ValueError("baseline version does not match the F4 record")
    bundle_model = bundle_loader(Path(bundle_dir))
    if bundle_model.bundle.manifest_sha256 != expected_bundle_sha:
        raise ValueError("ModelBundle changed between verification and model loading")
    return FrozenArtifacts(
        bundle_model, baseline_model, expected_bundle_sha, actual_baseline_sha
    )


def verify_f5_artifacts(
    bundle_dir: str | Path,
    f5_record: str | Path,
    *,
    bundle_verifier: Callable[[Path], Any],
    bundle_loader: Callable[[Path], Any],
) -> tuple[Any, str]:
    """Verify a frozen v2 bundle against its F5 record before it joins the D48 run set.

    F5's format-specific verifier and architecture-dispatching loader are deliberately injected:
    they must be supplied by the reviewed F5 implementation, never guessed by this older F4 loader.
    """
    if not callable(bundle_verifier) or not callable(bundle_loader):
        raise TypeError("F5 bundle verifier and architecture-aware loader are required")
    try:
        record = json.loads(Path(f5_record).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read F5 record: {exc}") from exc
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("checks"), list)
        or not record["checks"]
    ):
        raise TypeError("F5 record is missing its checks")
    if not all(
        isinstance(item, dict) and item.get("ok") is True for item in record["checks"]
    ):
        raise ValueError("F5 record contains a failed or malformed check")
    expected_sha = record.get("manifest_sha256")
    expected_version = record.get("bundle_version")
    if not isinstance(expected_sha, str) or not _SHA256.fullmatch(expected_sha):
        raise ValueError("F5 record has no valid manifest SHA-256")
    if not isinstance(expected_version, str) or not expected_version.strip():
        raise ValueError("F5 record has no bundle version")

    directory = Path(bundle_dir)
    verified = bundle_verifier(directory)
    if getattr(verified, "manifest_sha256", None) != expected_sha:
        raise ValueError("v2 bundle manifest does not match the F5 record")
    manifest = getattr(verified, "manifest", None)
    if getattr(manifest, "bundle_version", None) != expected_version:
        raise ValueError("v2 bundle version does not match the F5 record")
    model = bundle_loader(directory)
    loaded_bundle = getattr(model, "bundle", None)
    if getattr(loaded_bundle, "manifest_sha256", None) != expected_sha:
        raise ValueError("v2 bundle changed between verification and model loading")
    loaded_manifest = getattr(loaded_bundle, "manifest", None)
    if getattr(loaded_manifest, "bundle_version", None) != expected_version:
        raise ValueError("loaded v2 model version does not match the F5 record")
    return model, expected_sha


def _require_cpu_eval_model(model: Any, *, name: str) -> Any:
    import torch

    if not hasattr(model, "parameters") or not callable(model.parameters):
        raise TypeError(f"{name} loaded model must expose parameters()")
    parameters = tuple(model.parameters())
    if not parameters or any(
        parameter.device.type != "cpu" for parameter in parameters
    ):
        raise ValueError(f"{name} model must be loaded on CPU")
    if getattr(model, "training", None) is not False:
        raise ValueError(f"{name} model must be in evaluation mode")
    return torch


def _raw_logit_scorer(
    model: Any, feature_builder: Callable[[np.ndarray], np.ndarray], *, name: str
):
    torch = _require_cpu_eval_model(model, name=name)

    def score(waveform: np.ndarray) -> float:
        features = np.asarray(feature_builder(waveform), dtype=np.float32)
        if features.ndim != 3 or not np.isfinite(features).all():
            raise ValueError(
                f"{name} log-Mel features must be finite with shape (1, 64, 197)"
            )
        batch = torch.from_numpy(np.ascontiguousarray(features[None]))
        with torch.inference_mode():
            output = model(batch)
        values = np.asarray(output.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        if values.shape != (1,) or not np.isfinite(values[0]):
            raise ValueError(f"{name} model must return one finite logit per window")
        return float(values[0])

    return score


def build_frozen_scorers(
    artifacts: FrozenArtifacts,
) -> dict[str, Callable[[np.ndarray], Real]]:
    """Build every scoring closure from the already verified, in-memory frozen models."""
    from src.preprocessing.features import log_mel_spectrogram, mfcc_summary
    from src.training.normalization import apply_bin_stats

    bundle_model = artifacts.bundle_model
    bundle = bundle_model.bundle
    profile = bundle.profile
    if not isinstance(profile, PreprocessingProfile):
        raise TypeError(
            "verified F4 bundle must expose its parsed preprocessing profile"
        )

    def v1_features(waveform: np.ndarray) -> np.ndarray:
        mel = log_mel_spectrogram(waveform, profile)[None]
        return apply_bin_stats(mel, bundle_model.mean, bundle_model.std)[0]

    scorers: dict[str, Callable[[np.ndarray], Real]] = {
        "bundle-v1": _raw_logit_scorer(
            bundle_model.model, v1_features, name="bundle-v1"
        ),
    }
    baseline = artifacts.baseline_model

    def score_baseline(waveform: np.ndarray) -> float:
        values = np.asarray(
            baseline.score(mfcc_summary(waveform, profile)[None]), dtype=np.float64
        ).reshape(-1)
        if values.shape != (1,) or not np.isfinite(values[0]):
            raise ValueError("baseline-v1 must return one finite score per window")
        return float(values[0])

    scorers["baseline-v1"] = score_baseline

    if artifacts.v2_model is not None:
        v2 = artifacts.v2_model
        v2_bundle = getattr(v2, "bundle", None)
        metadata = getattr(v2, "metadata", None)
        model = getattr(v2, "model", None)
        if v2_bundle is None or not isinstance(metadata, Mapping) or model is None:
            raise TypeError(
                "verified F5 loader must expose bundle, model and parsed checkpoint metadata"
            )
        v2_profile = getattr(v2_bundle, "profile", None)
        if (
            not isinstance(v2_profile, PreprocessingProfile)
            or v2_profile.fingerprint() != profile.fingerprint()
        ):
            raise ValueError("F5 and F4 must use the same frozen preprocessing profile")
        architecture = metadata.get("architecture")
        if architecture in {"cnn-aug", "crnn"}:
            mean, std = getattr(v2, "mean", None), getattr(v2, "std", None)
            if mean is None or std is None:
                raise ValueError(
                    f"F5 {architecture} must load its verified log-Mel normalization statistics"
                )

            def v2_features(waveform: np.ndarray) -> np.ndarray:
                mel = log_mel_spectrogram(waveform, profile)[None]
                return apply_bin_stats(mel, mean, std)[0]

            scorers["bundle-v2"] = _raw_logit_scorer(
                model, v2_features, name="bundle-v2"
            )
        elif architecture == "aasist":
            torch = _require_cpu_eval_model(model, name="bundle-v2")
            if (
                getattr(v2, "mean", None) is not None
                or getattr(v2, "std", None) is not None
            ):
                raise ValueError(
                    "F5 aasist must not load log-Mel normalization statistics"
                )

            def score_aasist(waveform: np.ndarray) -> float:
                values = np.asarray(waveform, dtype=np.float32)
                if values.shape != (profile.samples,) or not np.isfinite(values).all():
                    raise ValueError(
                        "F5 aasist input must be one finite frozen-profile waveform"
                    )
                with torch.inference_mode():
                    output = model(torch.from_numpy(np.ascontiguousarray(values[None])))
                logits = np.asarray(
                    output.detach().cpu().numpy(), dtype=np.float64
                ).reshape(-1)
                if logits.shape != (1,) or not np.isfinite(logits[0]):
                    raise ValueError(
                        "F5 aasist must return one finite logit per window"
                    )
                return float(logits[0])

            scorers["bundle-v2"] = score_aasist
        else:
            raise ValueError(
                f"F5 checkpoint architecture is not approved by D49: {architecture!r}"
            )
    return scorers


def verify_pinned_manifest(path: str | Path, expected_sha256: str) -> str:
    """Hash the frozen metadata manifest before any external audio path can be opened."""
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        raise ValueError("expected manifest SHA-256 must be lowercase hexadecimal")
    manifest = Path(path)
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError("pinned manifest must be an existing regular file")
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"pinned manifest SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return actual


def verify_d58_rehearsal_report(
    path: str | Path,
    expected_sha256: str,
    *,
    baseline_artifact_sha256: str,
    profile_fingerprint: str,
    current_environment: Mapping[str, Any],
) -> str:
    """Verify a passed Stage 11 D58 rehearsal before the Stage 13 dataset claim.

    The report is the persisted result of D58's val-A/val-B comparison against the approved CPU
    MFCC references. Requiring its exact bytes, baseline identity, pinned reference hashes,
    tolerance results, and the T4/2026.10-compatible CPU scoring environment prevents a caller from
    proceeding with only the DEEP-VOICE manifest and frozen models in place.
    """
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        raise ValueError(
            "expected D58 rehearsal report SHA-256 must be lowercase hexadecimal"
        )
    if not isinstance(baseline_artifact_sha256, str) or not _SHA256.fullmatch(
        baseline_artifact_sha256
    ):
        raise ValueError("baseline artifact SHA-256 must be lowercase hexadecimal")
    if not isinstance(profile_fingerprint, str) or not _SHA256.fullmatch(
        profile_fingerprint
    ):
        raise ValueError("profile fingerprint must be lowercase SHA-256")
    report_path = Path(path)
    if report_path.is_symlink() or not report_path.is_file():
        raise ValueError("D58 rehearsal report must be an existing regular file")
    data = report_path.read_bytes()
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"D58 rehearsal report SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        report = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"D58 rehearsal report is not valid UTF-8 JSON: {exc}"
        ) from exc
    if (
        not isinstance(report, dict)
        or isinstance(report.get("schema_version"), bool)
        or report.get("schema_version") != 1
    ):
        raise ValueError("D58 rehearsal report must be a schema-version-1 object")
    if report.get("passed") is not True:
        raise ValueError("D58 rehearsal report is not passed")

    model_set = report.get("model_set")
    baseline_entry = (
        model_set.get("baseline-v1") if isinstance(model_set, dict) else None
    )
    if (
        not isinstance(baseline_entry, dict)
        or baseline_entry.get("artifact_sha256") != baseline_artifact_sha256
    ):
        raise ValueError(
            "D58 rehearsal baseline artifact does not match the verified Stage 13 baseline"
        )

    environment = report.get("environment")
    if not isinstance(environment, dict):
        raise TypeError("D58 rehearsal report has no environment record")
    packages = environment.get("packages")
    scoring = report.get("scoring")
    if (
        environment.get("python") != "3.13.16"
        or not isinstance(packages, dict)
        or packages.get("torch") != "2.11.0+cu130"
        or not isinstance(environment.get("gpu"), str)
        or "T4" not in environment["gpu"].upper()
        or environment.get("failures") != {}
        or not isinstance(scoring, dict)
        or scoring.get("device") != "cpu"
    ):
        raise ValueError(
            "D58 rehearsal must record the pinned 2026.10 Python/torch T4 runtime"
        )
    environment_fields = (
        "pip_freeze_sha256",
        "packages",
        "cpu",
        "gpu",
        "platform",
        "python",
        "failures",
    )
    if not isinstance(current_environment, Mapping) or any(
        current_environment.get(field) != environment.get(field)
        for field in environment_fields
    ):
        raise ValueError(
            "D58 rehearsal was not produced by the exact current runtime and T4 host"
        )

    def passed_record(records: Any, key: str) -> dict[str, Any]:
        item = records.get(key) if isinstance(records, dict) else None
        if not isinstance(item, dict) or item.get("passed") is not True:
            raise ValueError(f"D58 rehearsal is missing a passed check: {key}")
        return item

    checks = report.get("pre_claim_checks")
    baseline_check = passed_record(checks, "baseline.artifact_sha256")
    if baseline_check.get("actual") != baseline_artifact_sha256:
        raise ValueError(
            "D58 preflight baseline hash differs from the verified Stage 13 baseline"
        )
    profile_check = passed_record(checks, "profile_fingerprint")
    if profile_check.get("actual") != profile_fingerprint:
        raise ValueError(
            "D58 rehearsal profile differs from the verified F4 preprocessing profile"
        )

    def verify_reference_files(
        records: Any, key: str, split: str, expected_npz_sha256: str
    ) -> None:
        actual = passed_record(records, key).get("actual")
        expected_names = {f"{split}.npz", f"{split}.json"}
        if (
            not isinstance(actual, dict)
            or set(actual) != expected_names
            or actual.get(f"{split}.npz") != expected_npz_sha256
            or any(
                not isinstance(value, str) or not _SHA256.fullmatch(value)
                for value in actual.values()
            )
        ):
            raise ValueError(
                f"D58 {split} reference file hashes are missing or inconsistent"
            )

    # Stage 11's pre-claim reproduction gate covers val-A; the D58 rehearsal then covers both
    # val-A and val-B in known_answers below.
    reference_sha256 = D58_MFCC_REFERENCE_SHA256["val-A"]
    prefix = "reproduction.mfcc.val-A"
    reference_check = passed_record(checks, f"{prefix}.reference_sha256")
    if (
        reference_check.get("expected") != reference_sha256
        or reference_check.get("actual") != reference_sha256
    ):
        raise ValueError(
            "D58 val-A reference does not match the approved Stage 7 MFCC bytes"
        )
    verify_reference_files(
        checks, f"{prefix}.reference_files", "val-A", reference_sha256
    )
    passed_record(checks, f"{prefix}.reference_rows")
    tolerance = passed_record(checks, f"{prefix}.within_tolerance").get("actual")
    if (
        not isinstance(tolerance, dict)
        or tolerance.get("allclose") is not True
        or tolerance.get("rtol") != D58_MFCC_RTOL
        or tolerance.get("atol") != D58_MFCC_ATOL
    ):
        raise ValueError("D58 val-A MFCC comparison does not pass the frozen tolerance")

    answers = report.get("known_answers")
    for split, reference_sha256 in D58_MFCC_REFERENCE_SHA256.items():
        answer_prefix = f"{split}.mfcc"
        answer_reference = passed_record(answers, f"{answer_prefix}.reference_sha256")
        if (
            answer_reference.get("expected") != reference_sha256
            or answer_reference.get("actual") != reference_sha256
        ):
            raise ValueError(
                f"D58 {split} rehearsal reference does not match the approved Stage 7 bytes"
            )
        verify_reference_files(
            answers, f"{answer_prefix}.reference_files", split, reference_sha256
        )
        passed_record(answers, f"{answer_prefix}.reference_rows")
        answer_tolerance = passed_record(
            answers, f"{answer_prefix}.within_tolerance"
        ).get("actual")
        if (
            not isinstance(answer_tolerance, dict)
            or answer_tolerance.get("allclose") is not True
            or answer_tolerance.get("rtol") != D58_MFCC_RTOL
            or answer_tolerance.get("atol") != D58_MFCC_ATOL
        ):
            raise ValueError(
                f"D58 {split} rehearsal MFCC comparison does not pass the frozen tolerance"
            )
        prediction_check = passed_record(
            report.get("known_answers"), f"baseline.{split}.predictions_equal_reference"
        )
        if prediction_check.get("actual") != 0:
            raise ValueError(
                f"D58 {split} baseline predictions differ from the CPU reference"
            )
        max_difference = passed_record(
            answers, f"baseline.{split}.max_abs_score_diff"
        ).get("actual")
        near_threshold = passed_record(
            answers, f"baseline.{split}.near_threshold_clips"
        ).get("actual")
        if (
            isinstance(max_difference, bool)
            or not isinstance(max_difference, (int, float))
            or not math.isfinite(max_difference)
            or max_difference < 0
            or isinstance(near_threshold, bool)
            or not isinstance(near_threshold, int)
            or near_threshold < 0
        ):
            raise ValueError(
                f"D58 {split} baseline comparison diagnostics are malformed"
            )
    return actual_sha256


def load_external_manifest(
    path: str | Path, expected_sha256: str
) -> tuple[tuple[ManifestFile, ...], str]:
    """Verify and parse the frozen metadata-only inventory; never inspect audio bytes here."""
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        raise ValueError("expected manifest SHA-256 must be lowercase hexadecimal")
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("external manifest must be an existing regular file")
    data = source.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"pinned manifest SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    try:
        text = data.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        expected_columns = (
            "path",
            "label",
            "source_speaker",
            "target_speaker",
            "sample_rate",
            "channels",
            "duration_s",
            "hash",
        )
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise ValueError(f"manifest columns must be {expected_columns}")
        files = []
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"manifest row {line} has missing or extra columns")
            relpath = safe_relative_path(row["path"])
            if relpath != row["path"]:
                raise ValueError(f"manifest row {line} path is not canonical POSIX")
            label = row["label"]
            source_speaker = row["source_speaker"]
            target_speaker = row["target_speaker"] or None
            if label not in {"FAKE", "REAL"}:
                raise ValueError(f"manifest row {line} has invalid label {label!r}")
            if not source_speaker or source_speaker != source_speaker.strip():
                raise ValueError(f"manifest row {line} has invalid source speaker")
            if target_speaker is not None and target_speaker != target_speaker.strip():
                raise ValueError(f"manifest row {line} has invalid target speaker")
            try:
                sample_rate = int(row["sample_rate"])
                channels = int(row["channels"])
                duration = float(row["duration_s"])
            except ValueError:
                raise ValueError(
                    f"manifest row {line} has invalid audio metadata"
                ) from None
            if (
                sample_rate <= 0
                or channels <= 0
                or not math.isfinite(duration)
                or duration <= 0
            ):
                raise ValueError(f"manifest row {line} has non-positive audio metadata")
            digest = row["hash"]
            if not _SHA256.fullmatch(digest):
                raise ValueError(f"manifest row {line} has invalid file SHA-256")
            files.append(
                ManifestFile(
                    relpath,
                    label,
                    source_speaker,
                    target_speaker,
                    duration,
                    sample_rate,
                    channels,
                    digest,
                )
            )
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"external manifest is not valid UTF-8 CSV: {exc}") from exc

    if len(files) != 64 or len({item.file for item in files}) != 64:
        raise ValueError(
            f"frozen external manifest must contain 64 unique files, got {len(files)}"
        )
    counts = {
        label: sum(item.label == label for item in files) for label in ("REAL", "FAKE")
    }
    if counts != {"REAL": 8, "FAKE": 56}:
        raise ValueError(
            f"frozen external manifest label counts differ from the audit: {counts}"
        )
    if len({item.source_speaker for item in files}) != 8:
        raise ValueError(
            "frozen external manifest must contain the eight audited source speakers"
        )
    return tuple(files), actual


def prepare_guarded_run(
    config: Mapping[str, Any],
    *,
    run_id: str,
    bundle_verifier: Callable[[Path], Any] | None = None,
    bundle_loader: Callable[[Path], Any] | None = None,
    baseline_loader: Callable[[Path], Any] | None = None,
    baseline_hash: Callable[[Path], str] | None = None,
    f5_verifier: Callable[[Path], Any] | None = None,
    f5_loader: Callable[[Path], Any] | None = None,
) -> PreparedRun:
    """Verify the manifest and both F4 artifacts, then persist the one-shot claim.

    This is the required gate before a caller may open manifest-listed audio or invoke a scorer.
    Any failed check aborts before the claim; once claimed, any later failure consumes the run.
    """
    if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("run_id must be 32 lowercase hexadecimal characters")
    required = {
        "dataset",
        "registry_dir",
        "bundle_dir",
        "baseline_dir",
        "f4_record",
        "manifest_path",
        "manifest_sha256",
        "d58_rehearsal_report",
        "d58_rehearsal_report_sha256",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"prepared-run config is missing {missing}")
    if config["dataset"] != DATASET:
        raise ValueError(f"dataset must be {DATASET!r}")
    manifest_files, manifest_sha = load_external_manifest(
        config["manifest_path"], config["manifest_sha256"]
    )
    artifacts = verify_f4_artifacts(
        config["bundle_dir"],
        config["baseline_dir"],
        config["f4_record"],
        bundle_verifier=bundle_verifier,
        bundle_loader=bundle_loader,
        baseline_loader=baseline_loader,
        baseline_hash=baseline_hash,
    )
    v2_dir = config.get("v2_bundle_dir")
    f5_record = config.get("f5_record")
    if (v2_dir is None) != (f5_record is None):
        raise ValueError("v2_bundle_dir and f5_record must be supplied together")
    v2_model, v2_manifest_sha256 = None, None
    if v2_dir is None:
        if _tunis_now() <= D48_CUTOFF:
            raise RuntimeError(
                "D48 requires verified F5 before the cutoff; no claim was created"
            )
    else:
        if f5_verifier is None or f5_loader is None:
            raise RuntimeError(
                "F5 verification and architecture-aware model loading are not available"
            )
        v2_model, v2_manifest_sha256 = verify_f5_artifacts(
            v2_dir,
            f5_record,
            bundle_verifier=f5_verifier,
            bundle_loader=f5_loader,
        )
        artifacts = FrozenArtifacts(
            artifacts.bundle_model,
            artifacts.baseline_model,
            artifacts.bundle_manifest_sha256,
            artifacts.baseline_artifact_sha256,
            v2_model,
            v2_manifest_sha256,
        )
    # Stage 13's later T4 baseline pass must apply D58 (3) on this same runtime, not reuse a
    # rehearsal from a merely similar machine. The Stage 11 helper records environment only;
    # importing it does not read any dataset or audio.
    from src.evaluation.stage11_run import environment_record

    current_environment, _ = environment_record()
    d58_report_sha256 = verify_d58_rehearsal_report(
        config["d58_rehearsal_report"],
        config["d58_rehearsal_report_sha256"],
        baseline_artifact_sha256=artifacts.baseline_artifact_sha256,
        profile_fingerprint=artifacts.bundle_model.bundle.profile.fingerprint(),
        current_environment=current_environment,
    )
    scorers = build_frozen_scorers(artifacts)
    claim_path, claim_sha = claim_dataset_once(
        config["registry_dir"],
        dataset=DATASET,
        run_id=run_id,
        model_artifacts={
            "baseline-v1": artifacts.baseline_artifact_sha256,
            "bundle-v1": artifacts.bundle_manifest_sha256,
            "deepvoice-manifest": manifest_sha,
            "d58-rehearsal-report": d58_report_sha256,
            **({"bundle-v2": v2_manifest_sha256} if v2_manifest_sha256 else {}),
        },
    )
    model_set = (
        ("bundle-v1", "bundle-v2", "baseline-v1")
        if v2_model is not None
        else ("bundle-v1", "baseline-v1")
    )
    return PreparedRun(
        artifacts,
        manifest_files,
        manifest_sha,
        claim_path,
        claim_sha,
        d58_report_sha256,
        scorers,
        run_id,
        model_set,
    )


def _tunis_now() -> datetime:
    """Return local decision time for the D48 freeze cutoff."""
    return datetime.now(ZoneInfo("Africa/Tunis"))


def _claim_record_for_scoring(prepared: PreparedRun) -> dict[str, Any]:
    target = prepared.claim_path
    if target.is_symlink() or not target.is_file():
        raise ValueError("prepared dataset claim must be an existing regular file")
    data = target.read_bytes()
    if hashlib.sha256(data).hexdigest() != prepared.claim_sha256:
        raise ValueError("prepared dataset claim SHA-256 changed before scoring")
    try:
        record = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"prepared dataset claim is not valid JSON: {exc}") from exc
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != CLAIM_SCHEMA
        or isinstance(record.get("schema_version"), bool)
        or record.get("dataset") != DATASET
        or record.get("run_id") != prepared.run_id
        or record.get("status") != "started"
    ):
        raise ValueError("prepared dataset claim is not the expected started claim")
    artifact_hashes = {
        "baseline-v1": prepared.artifacts.baseline_artifact_sha256,
        "bundle-v1": prepared.artifacts.bundle_manifest_sha256,
        "deepvoice-manifest": prepared.manifest_sha256,
        "d58-rehearsal-report": prepared.d58_report_sha256,
    }
    if prepared.artifacts.v2_manifest_sha256 is not None:
        artifact_hashes["bundle-v2"] = prepared.artifacts.v2_manifest_sha256
    if record.get("model_artifacts") != dict(sorted(artifact_hashes.items())):
        raise ValueError("started claim does not match the prepared frozen artifacts")
    expected_set = (
        ("bundle-v1", "bundle-v2", "baseline-v1")
        if prepared.artifacts.v2_model
        else ("bundle-v1", "baseline-v1")
    )
    if prepared.model_set != expected_set:
        raise ValueError(
            "prepared model set does not match its verified frozen artifacts"
        )
    return record


def _manifest_audio_path(root: Path, relative_name: str) -> Path:
    """Resolve one canonical manifest path without following child symlinks outside the data root."""
    try:
        canonical = safe_relative_path(relative_name)
    except (TypeError, ValueError):
        raise ValueError("manifest audio path is not safe and relative") from None
    if canonical != relative_name:
        raise ValueError("manifest audio path must use canonical POSIX separators")
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError("audio root must be an existing directory")
    candidate = resolved_root
    for part in PurePosixPath(canonical).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"manifest audio path contains a symlink: {relative_name}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        raise ValueError(
            f"manifest audio path escapes the data root: {relative_name}"
        ) from None
    if not resolved.is_file():
        raise ValueError(f"manifest audio path is not a regular file: {relative_name}")
    return resolved


def _decode_audio_bytes(data: bytes, name: str) -> tuple[np.ndarray, int]:
    """Decode bytes only after their manifest hash is verified; preserve source channel layout."""
    import soundfile as sf

    try:
        waveform, sample_rate = sf.read(
            io.BytesIO(data), dtype="float32", always_2d=False
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"cannot decode manifest audio {name!r}: {exc}") from exc
    return np.asarray(waveform), int(sample_rate)


def score_guarded_dataset(
    prepared: PreparedRun,
    audio_root: str | Path,
    *,
    output_dir: str | Path,
    decoder: Callable[[bytes, str], tuple[np.ndarray, int]] | None = None,
    n_resamples: int = 2000,
    seed: int = 2026,
) -> GuardedRunResult:
    """Consume a prepared claim exactly once, finalizing any post-claim error as incomplete."""
    if not isinstance(prepared, PreparedRun):
        raise TypeError("prepared must be returned by prepare_guarded_run")
    try:
        _claim_record_for_scoring(prepared)
        return _score_guarded_dataset_impl(
            prepared,
            audio_root,
            output_dir=output_dir,
            decoder=decoder,
            n_resamples=n_resamples,
            seed=seed,
        )
    except BaseException as exc:
        try:
            finalize_claim(
                prepared.claim_path,
                run_id=prepared.run_id,
                status="incomplete",
                detail=f"{type(exc).__name__}: {str(exc)[:500]}",
            )
        except Exception as finalize_error:  # noqa: BLE001 - keep the original scoring failure
            if hasattr(exc, "add_note"):
                exc.add_note(
                    "Claim finalization also failed: "
                    f"{type(finalize_error).__name__}: {finalize_error}"
                )
        raise


def _score_guarded_dataset_impl(
    prepared: PreparedRun,
    audio_root: str | Path,
    *,
    output_dir: str | Path,
    decoder: Callable[[bytes, str], tuple[np.ndarray, int]] | None = None,
    n_resamples: int = 2000,
    seed: int = 2026,
) -> GuardedRunResult:
    """Score all prepared models in one guarded pass and persist scores plus report.

    The caller must first obtain `PreparedRun` from the frozen-artifact and D48 preflight. Each
    manifest file is read once as bytes, hash-checked before decode, and then split into complete
    protocol windows. Every model consumes the same Stage 6-preprocessed window. Any error after
    the one-shot claim is recorded as `incomplete`; this function never retries or resumes.
    """
    if not isinstance(prepared, PreparedRun):
        raise TypeError("prepared must be returned by prepare_guarded_run")
    run_id = prepared.run_id
    scorers = prepared.scorers
    if not isinstance(scorers, Mapping) or set(scorers) != set(prepared.model_set):
        raise ValueError("prepared scorers must exactly match the verified model set")
    if any(not callable(score) for score in scorers.values()):
        raise TypeError("prepared model scorers must be callable")
    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, int)
        or n_resamples < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError("n_resamples must be positive and seed must be an integer")

    bundle_model = prepared.artifacts.bundle_model
    bundle = getattr(bundle_model, "bundle", None)
    profile = getattr(bundle, "profile", None)
    policy = getattr(bundle, "policy", None)
    if not isinstance(profile, PreprocessingProfile) or policy is None:
        raise ValueError(
            "verified F4 must provide its parsed preprocessing profile and risk policy"
        )
    model_context: dict[str, dict[str, Any]] = {}
    model_hashes = {
        "bundle-v1": prepared.artifacts.bundle_manifest_sha256,
        "baseline-v1": prepared.artifacts.baseline_artifact_sha256,
    }
    model_context["bundle-v1"] = {
        "role": "cnn_bundle",
        "bundle_version": bundle.manifest.bundle_version,
        "policy": policy,
        "threshold": policy.threshold,
        "half_width": policy.half_width,
    }
    baseline = prepared.artifacts.baseline_model
    model_context["baseline-v1"] = {
        "role": "baseline",
        "bundle_version": None,
        "policy": None,
        "threshold": baseline.threshold,
        "half_width": 0.0,
    }
    if "bundle-v2" in prepared.model_set:
        v2 = prepared.artifacts.v2_model
        v2_bundle = getattr(v2, "bundle", None)
        v2_policy = getattr(v2_bundle, "policy", None)
        if (
            v2_bundle is None
            or v2_policy is None
            or prepared.artifacts.v2_manifest_sha256 is None
        ):
            raise ValueError(
                "prepared F5 model must provide its verified bundle, policy and manifest hash"
            )
        model_hashes["bundle-v2"] = prepared.artifacts.v2_manifest_sha256
        model_context["bundle-v2"] = {
            "role": "cnn_bundle",
            "bundle_version": v2_bundle.manifest.bundle_version,
            "policy": v2_policy,
            "threshold": v2_policy.threshold,
            "half_width": v2_policy.half_width,
        }
    if set(model_context) != set(prepared.model_set):
        raise ValueError("prepared models do not have complete scoring metadata")

    target_root = Path(audio_root)
    target_output = Path(output_dir)
    decode = decoder or _decode_audio_bytes
    rows_by_model = {model_id: [] for model_id in prepared.model_set}
    file_accounting: dict[str, dict[str, int]] = {}
    total_windows = total_dropped_samples = total_filter_dropped = 0
    try:
        _claim_record_for_scoring(prepared)
        for item in sorted(prepared.manifest_files, key=lambda entry: entry.file):
            if not isinstance(item.sha256, str) or not _SHA256.fullmatch(item.sha256):
                raise ValueError(
                    f"manifest file {item.file!r} has no valid audio SHA-256"
                )
            if item.sample_rate is None or item.channels is None:
                raise ValueError(
                    f"manifest file {item.file!r} lacks sample-rate or channel metadata"
                )
            path = _manifest_audio_path(target_root, item.file)
            data = path.read_bytes()
            actual_sha = hashlib.sha256(data).hexdigest()
            if actual_sha != item.sha256:
                raise ValueError(f"manifest audio SHA-256 mismatch for {item.file!r}")
            waveform, sample_rate = decode(data, item.file)
            waveform = np.asarray(waveform)
            if waveform.ndim not in (1, 2) or waveform.size == 0:
                raise ValueError(
                    f"decoded audio {item.file!r} must be a non-empty mono/stereo waveform"
                )
            channels = 1 if waveform.ndim == 1 else waveform.shape[-1]
            if sample_rate != item.sample_rate or channels != item.channels:
                raise ValueError(
                    f"decoded audio metadata mismatch for {item.file!r}: "
                    f"sample_rate={sample_rate}, channels={channels}"
                )
            sample_axis = 0
            expected_windows = math.floor(
                item.duration_seconds / float(profile.waveform["duration_seconds"])
            )
            observed_windows, _ = _window_count(
                waveform.shape[sample_axis], sample_rate, profile
            )
            if observed_windows != expected_windows:
                raise ValueError(
                    f"decoded window count differs from manifest duration for {item.file!r}: "
                    f"expected {expected_windows}, got {observed_windows}"
                )

            window_runs = score_window_models(waveform, sample_rate, profile, scorers)
            file_accounting[item.file] = {
                "complete_windows": next(iter(window_runs.values())).n_complete_windows,
                "dropped_samples": next(iter(window_runs.values())).n_dropped_samples,
                "filter_dropped_windows": next(
                    iter(window_runs.values())
                ).n_filter_dropped,
            }
            total_windows += file_accounting[item.file]["complete_windows"]
            total_dropped_samples += file_accounting[item.file]["dropped_samples"]
            total_filter_dropped += file_accounting[item.file]["filter_dropped_windows"]
            for model_id, window_run in window_runs.items():
                context = model_context[model_id]
                rows_by_model[model_id].extend(
                    rows_from_window_run(
                        window_run,
                        file=item.file,
                        source_speaker=item.source_speaker,
                        target_speaker=item.target_speaker,
                        label=item.label,
                        run_id=run_id,
                        model_role=context["role"],
                        bundle_version=context["bundle_version"],
                        policy=context["policy"],
                        baseline_threshold=(
                            context["threshold"]
                            if context["role"] == "baseline"
                            else None
                        ),
                    )
                )

        if sum(item.source_speaker is None for item in prepared.manifest_files):
            raise ValueError("manifest source speaker is required for every file")
        for model_id, rows in rows_by_model.items():
            _validate_complete_external_rows(rows, prepared.manifest_files)
        score_paths = {}
        for model_id in prepared.model_set:
            context = model_context[model_id]
            score_paths[model_id] = write_external_score_file(
                target_output,
                model_id=model_id,
                model_role=context["role"],
                model_artifact_sha256=model_hashes[model_id],
                manifest_sha256=prepared.manifest_sha256,
                rows=rows_by_model[model_id],
            )
        stored = {
            model_id: read_external_score_file(*paths)
            for model_id, paths in score_paths.items()
        }
        report: dict[str, Any] = {
            "schema_version": 1,
            "dataset": DATASET,
            "run_id": run_id,
            "manifest_sha256": prepared.manifest_sha256,
            "d58_rehearsal_report_sha256": prepared.d58_report_sha256,
            "model_set": list(prepared.model_set),
            "files_scored": len(file_accounting),
            "complete_windows": total_windows,
            "dropped_samples": total_dropped_samples,
            "filter_dropped_windows": total_filter_dropped,
            "files": file_accounting,
            "score_outputs": {},
            "speaker_views": {},
            "file_views": {},
            "paired_eer": {},
            "hypotheses": {},
        }
        for model_id, (sidecar, stored_rows) in stored.items():
            context = model_context[model_id]
            report["score_outputs"][model_id] = {
                "csv": score_paths[model_id][0].name,
                "sidecar": score_paths[model_id][1].name,
                "csv_sha256": sidecar["csv_sha256"],
                "n_rows": len(stored_rows),
            }
            report["speaker_views"][model_id] = {}
            for view in ("filtered", "unfiltered"):
                report["speaker_views"][model_id][view] = speaker_level_view(
                    stored_rows,
                    prepared.manifest_files,
                    model_role=context["role"],
                    threshold=context["threshold"],
                    half_width=context["half_width"],
                    view=view,
                )
                if context["role"] == "cnn_bundle":
                    report["file_views"].setdefault(model_id, {})[view] = (
                        file_level_view(
                            stored_rows,
                            prepared.manifest_files,
                            threshold=context["threshold"],
                            half_width=context["half_width"],
                            view=view,
                            n_resamples=n_resamples,
                            seed=seed,
                        )
                    )
        for model_id in prepared.model_set:
            if model_context[model_id]["role"] != "cnn_bundle":
                continue
            comparison_id = f"{model_id}-minus-baseline-v1"
            report["paired_eer"][comparison_id] = paired_eer_comparison(
                stored[model_id],
                stored["baseline-v1"],
                prepared.manifest_files,
                view="filtered",
                n_resamples=n_resamples,
                seed=seed,
            )
        if "bundle-v2" in prepared.model_set:
            # H8 is pre-registered as EER(v1) - EER(v2), filtered primary and unfiltered sensitivity.
            h8_views = {}
            for view in ("filtered", "unfiltered"):
                paired = paired_eer_comparison(
                    stored["bundle-v1"],
                    stored["bundle-v2"],
                    prepared.manifest_files,
                    view=view,
                    n_resamples=n_resamples,
                    seed=seed,
                    second_model_role="cnn_bundle",
                )
                supported = (
                    None
                    if paired.difference is None or paired.interval is None
                    else paired.difference > 0 and paired.interval.low > 0
                )
                h8_views[view] = {
                    "view": paired.view,
                    "n_windows": paired.n_windows,
                    "n_speakers": paired.n_speakers,
                    "v1_eer": paired.cnn_eer,
                    "v2_eer": paired.baseline_eer,
                    "difference_v1_minus_v2": paired.difference,
                    "interval_95": paired.interval,
                    "n_resamples": paired.n_resamples,
                    "n_invalid": paired.n_invalid,
                    "supported": supported,
                    "undefined_reason": paired.undefined_reason,
                }
            report["hypotheses"]["H8"] = {
                "status": "evaluated",
                "primary_view": "filtered",
                **h8_views,
            }
        else:
            report["hypotheses"]["H8"] = {
                "status": "not_tested_cutoff_excluded_v2",
                "reason": "D48 cutoff excluded v2 from the single Stage 13 run",
            }
        report_path, report_sha = write_external_report(
            target_output / "report.json", report
        )
        claim_sha = finalize_claim(
            prepared.claim_path, run_id=run_id, status="succeeded"
        )
        claim_bytes = prepared.claim_path.read_bytes()
        return GuardedRunResult(
            score_paths,
            report_path,
            report_sha,
            prepared.claim_path,
            claim_sha,
            len(claim_bytes.splitlines()),
            len(file_accounting),
            total_windows,
            total_dropped_samples,
            total_filter_dropped,
        )
    except BaseException:  # noqa: TRY203 - preserve exception after wrapper-level finalization
        raise


def _claim_path(registry_dir: Path, dataset: str) -> Path:
    if dataset != DATASET:
        raise ValueError(f"dataset must be exactly {DATASET!r}")
    key = hashlib.sha256(dataset.encode("utf-8")).hexdigest()
    return registry_dir / f"claim-{key}.json"


def claim_dataset_once(
    registry_dir: str | Path,
    *,
    dataset: str,
    run_id: str,
    model_artifacts: Mapping[str, str],
) -> tuple[Path, str]:
    """Durably create the dataset-wide claim before any audio read; never replace a claim.

    The caller must point registry_dir at the reviewed durable location (for the real run, Drive).
    An O_EXCL create makes competing starts race safely. A crash leaves the claim in place and
    consumes the one run; it is deliberately not resumable.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be non-empty")
    if not isinstance(model_artifacts, Mapping) or not model_artifacts:
        raise ValueError(
            "model_artifacts must contain at least one model-set fingerprint"
        )
    for name, digest in model_artifacts.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
        ):
            raise ValueError(
                "model_artifacts must map non-empty names to lowercase SHA-256 values"
            )
    directory = Path(registry_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = _claim_path(directory, dataset)
    payload = {
        "schema_version": CLAIM_SCHEMA,
        "dataset": dataset,
        "run_id": run_id,
        "status": "started",
        "claimed_at": datetime.now(UTC).isoformat(),
        "model_artifacts": dict(sorted(model_artifacts.items())),
    }
    data = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"single-run claim already exists for dataset {dataset!r}; do not retry"
        ) from exc
    # If writing fails, leave the file in place: an incomplete claim still consumes the run.
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    verified = target.read_bytes()
    if verified != data:
        raise OSError(
            "durable claim read-back did not match the bytes written; abort before scoring"
        )
    return target, hashlib.sha256(verified).hexdigest()


def finalize_claim(
    path: str | Path, *, run_id: str, status: str, detail: str = ""
) -> str:
    """Record success/failure after a run without making the claim reusable."""
    if status not in {"succeeded", "failed", "incomplete"}:
        raise ValueError("status must be succeeded, failed, or incomplete")
    target = Path(path)
    record = json.loads(target.read_text(encoding="utf-8"))
    if record.get("run_id") != run_id or record.get("status") != "started":
        raise RuntimeError("claim identity or state mismatch")
    record["status"] = status
    record["finished_at"] = datetime.now(UTC).isoformat()
    if detail:
        record["detail"] = detail
    data = (
        json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    fd, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        verified = target.read_bytes()
        if verified != data:
            raise OSError("claim finalization read-back mismatch")
        return hashlib.sha256(verified).hexdigest()
    finally:
        Path(temporary).unlink(missing_ok=True)
