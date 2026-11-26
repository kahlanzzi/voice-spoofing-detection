from pathlib import Path

import pytest

from src.evaluation.live_warmup_plan import build_call_plan
from src.preprocessing.profile import PreprocessingError


MANIFEST = Path(__file__).resolve().parents[2] / "data/manifests/for2sec_frozen_manifest.csv"


def test_plan_uses_only_verified_val_b_without_loss_or_replacement():
    plan = build_call_plan(MANIFEST)
    clips = {row["path"]: row for row in plan["clips"]}
    assert clips and all(row["split"] == "val-B" for row in clips.values())
    used = []
    for label, group in plan["plans"].items():
        assert group["calls"]
        for call in group["calls"]:
            assert len(call) == 15
            assert all(clips[path]["label"] == label for path in call)
            used.extend(call)
        assert len(group["leftovers"]) < 15
        assert all(clips[path]["label"] == label for path in group["leftovers"])
        used.extend(group["leftovers"])
    assert len(used) == len(set(used)) == len(clips)
    assert set(used) == set(clips)
    assert plan == build_call_plan(MANIFEST)


def test_modified_manifest_is_rejected_before_planning(tmp_path):
    altered = tmp_path / "manifest.csv"
    altered.write_bytes(MANIFEST.read_bytes().replace(b"val-B", b"val-A", 1))
    with pytest.raises(PreprocessingError, match="SHA-256 mismatch"):
        build_call_plan(altered)
