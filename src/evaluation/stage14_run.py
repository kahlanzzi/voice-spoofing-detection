"""Fail-closed Stage 14 v1 runner gates (D3, D41, D58; implementation under review).

This first review unit contains the clean-score hand-over gate only.  The scoring phases will be
added here before any FoR test audio is opened: one purpose-keyed claim per v1 model set, degraded
conditions only, unchanged F3 preprocessing, D58 before baseline scoring, and no DEEP-VOICE path.
"""
from __future__ import annotations

from pathlib import Path

from src.evaluation.clip_scores import ClipScores, read_clip_scores
from src.evaluation.robustness import RobustnessError
from src.evaluation.run_registry import COMPLETED, RunRegistry
from src.preprocessing.validation import F2_MANIFEST_SHA256, sha256_file


def read_stage11_clean_scores(
    score_dir: str | Path,
    *,
    model_id: str,
    registry_path: str | Path,
    manifest_path: str | Path,
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
    if meta["model_role"] not in {"baseline", "cnn_bundle_seed"}:
        raise RobustnessError(
            "Stage 14 v1 accepts only baseline or frozen CNN bundle clean scores, got "
            f"{meta['model_role']!r}")
    matches = [
        record for record in RunRegistry(registry_path).records()
        if record.status == COMPLETED
        and record.run_id == meta["run_id"]
        and record.dataset == "for"
        and record.split == "test"
        and record.evaluation_purpose == "stage11_test"
        and model_id in record.model_set
    ]
    if len(matches) != 1:
        raise RobustnessError(
            f"clean score run_id {meta['run_id']} has {len(matches)} matching completed "
            "Stage 11 registry records; expected exactly one")
    return scores
