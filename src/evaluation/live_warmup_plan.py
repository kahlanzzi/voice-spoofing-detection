"""Metadata-only D52 call membership from the verified F2 manifest.

No audio is decoded, no scores are read, and no registry claim is created.
"""
from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re

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


def export_call_plan(manifest_path: Path, output: Path, *, code_commit: str) -> str:
    """Write once; return the digest of the exact exported bytes.

    The caller records the clean implementation commit. This command does not
    assert that a supplied commit is reviewed or authorize scoring.
    """
    if not isinstance(code_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise ValueError("code_commit must be a full lowercase Git SHA")
    plan = build_call_plan(manifest_path)
    plan["code_commit"] = code_commit
    plan["python_version"] = platform.python_version()
    payload = (json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with Path(output).open("xb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args(argv)
    digest = export_call_plan(args.manifest, args.output, code_commit=args.code_commit)
    print(f"call_plan_sha256: {digest}")
    print(f"call_plan: {args.output}")


if __name__ == "__main__":
    main()
