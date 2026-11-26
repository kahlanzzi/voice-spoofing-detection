"""Metadata-only D52 call membership from the verified F2 manifest.

No audio is decoded, no scores are read, and no registry claim is created.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from src.evaluation.live_warmup import plan_same_label_calls
from src.preprocessing.validation import F2_MANIFEST_SHA256, load_f2_manifest


def build_call_plan(manifest_path: Path) -> dict:
    """Verify F2 before selecting val-B; retain labels, group and sample hashes."""
    manifest = load_f2_manifest(manifest_path)
    rows = manifest.loc[manifest["split"] == "val-B"].sort_values("path")
    if set(rows["label"]) != {"REAL", "FAKE"}:
        raise ValueError("val-B must contain both REAL and FAKE")
    plans = {}
    for label in ("REAL", "FAKE"):
        selected = rows.loc[rows["label"] == label]
        plan = plan_same_label_calls(selected["path"])
        if not plan.calls:
            raise ValueError(f"val-B {label} has no complete fifteen-clip call")
        plans[label] = asdict(plan)
    return {
        "schema_version": 1,
        "manifest_sha256": F2_MANIFEST_SHA256,
        "split": "val-B",
        "seed": 2026,
        "rng": "Python random.Random / MT19937",
        "slots_per_call": 15,
        "slot_seconds": 2,
        "replacement": False,
        "plans": plans,
        "clips": rows.to_dict(orient="records"),
    }
