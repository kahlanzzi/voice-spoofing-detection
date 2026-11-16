"""The two v2 roles of the per-clip score files (PR V4b; D48 (7), D45): synthetic rows only.

v2_bundle_seed / v2_other_seed have exactly the applicability of cnn_bundle_seed / cnn_other_seed;
the v1 table (APPLICABLE_COLUMNS, pinned in test_clip_scores.py) is unchanged.
"""
from __future__ import annotations

import pytest

from src.evaluation.clip_scores import (
    APPLICABLE_COLUMNS,
    MODEL_ROLES,
    PREDICTIONS,
    ROLE_COLUMNS,
    V2_APPLICABLE_COLUMNS,
    read_clip_scores,
    write_clip_scores,
)

CLIPS = ("synth/fake/f1.wav", "synth/real/r1.wav", "synth/fake/f2.wav")
LABELS = ("FAKE", "REAL", "FAKE")
GROUPS = ("g1", "g2", "g1")
RAW = (1.5, -2.25, 0.1)
P = (0.8, 0.1, 0.52)
PRED = ("synthetic", "authentic", "uncertain")


def _write(folder, role, **change):
    args = dict(evaluation_purpose="stage11_test", run_id="0" * 32, model_id=f"crnn-v2-{role}",
                model_role=role, model_artifact_sha256="a" * 64, manifest_sha256="b" * 64, clip_id=CLIPS,
                label=LABELS, group_id=GROUPS, raw_score=RAW)
    if role == "v2_bundle_seed":
        args.update(p=P, prediction=PRED)
    args.update(change)
    return write_clip_scores(folder, **args)


def test_v2_roles_mirror_the_v1_cnn_roles_and_leave_the_v1_table_unchanged():
    assert V2_APPLICABLE_COLUMNS == {"v2_bundle_seed": APPLICABLE_COLUMNS["cnn_bundle_seed"],
                                     "v2_other_seed": APPLICABLE_COLUMNS["cnn_other_seed"]}
    assert set(APPLICABLE_COLUMNS) == {"baseline", "cnn_bundle_seed", "cnn_other_seed"}
    assert ROLE_COLUMNS == {**APPLICABLE_COLUMNS, **V2_APPLICABLE_COLUMNS}
    assert MODEL_ROLES == frozenset(ROLE_COLUMNS)
    assert PREDICTIONS["v2_bundle_seed"] == PREDICTIONS["cnn_bundle_seed"]
    assert "v2_other_seed" not in PREDICTIONS


@pytest.mark.parametrize("role", ["v2_bundle_seed", "v2_other_seed"])
def test_v2_role_round_trip(tmp_path, role):
    _write(tmp_path, role)
    got = read_clip_scores(tmp_path, f"crnn-v2-{role}")
    assert got.metadata["model_role"] == role
    assert got.metadata["applicable_columns"] == list(V2_APPLICABLE_COLUMNS[role])
    assert got.clip_id == tuple(sorted(CLIPS)) and got.raw_score.tolist() == [
        dict(zip(CLIPS, RAW))[c] for c in sorted(CLIPS)]
    if role == "v2_bundle_seed":
        assert got.p.tolist() == [dict(zip(CLIPS, P))[c] for c in sorted(CLIPS)]
        assert got.prediction == tuple(dict(zip(CLIPS, PRED))[c] for c in sorted(CLIPS))
    else:
        assert got.p is None and got.prediction is None


@pytest.mark.parametrize("change, match", [
    ({"p": P}, "p is not applicable for model_role 'v2_other_seed'"),
    ({"prediction": PRED}, "prediction is not applicable for model_role 'v2_other_seed'"),
])
def test_v2_other_seed_refuses_p_and_prediction(tmp_path, change, match):
    with pytest.raises(ValueError, match=match):
        _write(tmp_path, "v2_other_seed", **change)
    assert list(tmp_path.iterdir()) == []


def test_v2_bundle_seed_requires_p_and_a_three_way_prediction(tmp_path):
    with pytest.raises(ValueError, match="p is applicable for model_role 'v2_bundle_seed'"):
        _write(tmp_path, "v2_bundle_seed", p=None)
    with pytest.raises(ValueError, match="prediction"):
        _write(tmp_path, "v2_bundle_seed", prediction=("FAKE",) + PRED[1:])
    assert list(tmp_path.iterdir()) == []
