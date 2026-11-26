"""Descriptive D52 reports from calibrated slot probabilities; no audio access."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re

from src.evaluation.live_warmup import warmup_checkpoints
from src.evaluation.metrics import binary_metrics, eer, roc_auc, three_way_metrics
from src.evaluation.live_warmup_plan import build_call_plan
from src.risk.decisions import three_way_decisions


def checkpoint_report(calls, *, threshold: float, half_width: float) -> dict:
    """Report fixed checkpoints with explicit denominators and missingness.

    Each call has a unique call_id, label REAL/FAKE, and fifteen calibrated p
    values (None means D17 no speech). Caller verifies provenance and membership.
    This is descriptive val-B evidence, not a held-out performance estimate.
    """
    three_way_decisions([0.5], threshold, half_width)
    records = []
    seen = set()
    for call in calls:
        if set(call) != {"call_id", "label", "probabilities"}:
            raise ValueError("call fields must be call_id, label and probabilities")
        identifier = call["call_id"]
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise ValueError("call_id must be unique and nonempty")
        seen.add(identifier)
        if call["label"] not in ("REAL", "FAKE"):
            raise ValueError("call label must be REAL or FAKE")
        checkpoints = warmup_checkpoints(
            call["probabilities"], threshold=threshold, half_width=half_width,
        )
        records.append({"call_id": identifier, "label": call["label"],
                        "checkpoints": [asdict(item) for item in checkpoints]})
    if not records:
        raise ValueError("at least one simulated call is required")
    summaries = []
    for index, count in enumerate((3, 5, 10)):
        reached = [row for row in records if row["checkpoints"][index]["reached"]]
        labels = [int(row["label"] == "FAKE") for row in reached]
        scores = [row["checkpoints"][index]["mean_p"] for row in reached]
        predictions = [row["checkpoints"][index]["prediction"] for row in reached]
        both = len(set(labels)) == 2
        correct = sum(pred == ("synthetic" if label else "authentic")
                      for label, pred in zip(labels, predictions))
        summaries.append({
            "kept_windows": count, "total_calls": len(records),
            "reached": len(reached), "unreached": len(records) - len(reached),
            "reached_by_label": {label: sum(row["label"] == label for row in reached)
                                 for label in ("REAL", "FAKE")},
            "correct_verdicts": correct,
            "verdict_accuracy_among_reached": correct / len(reached) if reached else None,
            "forced_binary": asdict(binary_metrics(labels, scores, threshold)) if reached else None,
            "three_way": asdict(three_way_metrics(labels, predictions)) if reached else None,
            "eer": eer(labels, scores)[0] if both else None,
            "roc_auc": roc_auc(labels, scores) if both else None,
            "score_metric_unavailable_reason": None if both else "fewer than two reached classes",
        })
    return {"schema_version": 1, "scope": "descriptive val-B warm-up",
            "threshold": threshold, "half_width": half_width,
            "accuracy_denominator": "reached calls; uncertain counts as an incorrect verdict",
            "unreached_rule": "excluded from metrics and counted separately, never a deferral",
            "checkpoints": summaries, "calls": records}


def report_from_plan(plan_path: Path, manifest_path: Path, probabilities: dict, *,
                     expected_plan_sha256: str, threshold: float, half_width: float) -> dict:
    """Bind slot scores to exact exported membership, verified again against F2.

    Scores must map every planned clip ID (excluding leftovers) to p or None.
    Model identity/calibration and D17 provenance remain the scoring caller's duty.
    """
    if not isinstance(expected_plan_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_plan_sha256):
        raise ValueError("expected_plan_sha256 must be a lowercase SHA-256")
    data = Path(plan_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_plan_sha256:
        raise ValueError("call-plan SHA-256 mismatch")
    plan = json.loads(data)
    expected = build_call_plan(manifest_path)
    expected = json.loads(json.dumps(expected))  # tuple/list normalization
    if not isinstance(plan, dict) or set(plan) != set(expected) | {"code_commit", "python_version"}:
        raise ValueError("call-plan schema mismatch")
    if any(plan[key] != value for key, value in expected.items()):
        raise ValueError("call-plan membership differs from verified val-B plan")
    if not isinstance(plan["code_commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", plan["code_commit"]):
        raise ValueError("invalid call-plan code commit")
    if not isinstance(plan["python_version"], str) or not plan["python_version"].strip():
        raise ValueError("missing call-plan Python version")
    required = {path for group in plan["plans"].values() for call in group["calls"] for path in call}
    if not isinstance(probabilities, dict) or set(probabilities) != required:
        raise ValueError("scores must match planned clip membership exactly; no missing or extra IDs")
    calls = []
    for label, group in plan["plans"].items():
        for index, members in enumerate(group["calls"]):
            calls.append({"call_id": f"{label}-{index:04d}", "label": label,
                          "probabilities": [probabilities[path] for path in members]})
    result = checkpoint_report(calls, threshold=threshold, half_width=half_width)
    result["call_plan_sha256"] = expected_plan_sha256
    result["manifest_sha256"] = plan["manifest_sha256"]
    result["leftover_counts"] = {label: len(group["leftovers"]) for label, group in plan["plans"].items()}
    return result


def report_json(report: dict) -> str:
    """Serialize undefined metric rates as null, never nonstandard JSON NaN."""
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value
    return json.dumps(clean(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
