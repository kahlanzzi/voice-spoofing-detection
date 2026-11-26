"""Descriptive D52 reports from calibrated slot probabilities; no audio access."""
from __future__ import annotations

from dataclasses import asdict

from src.evaluation.live_warmup import warmup_checkpoints
from src.evaluation.metrics import binary_metrics, eer, roc_auc, three_way_metrics
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
