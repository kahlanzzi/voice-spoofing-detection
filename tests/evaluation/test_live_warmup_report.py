import pytest
import hashlib
import json
from pathlib import Path

from src.evaluation.live_warmup_report import checkpoint_report, report_from_plan, report_json
from src.evaluation.live_warmup_plan import export_call_plan


def call(identifier, label, probabilities):
    return dict(call_id=identifier, label=label, probabilities=probabilities)


def report(calls):
    return checkpoint_report(calls, threshold=0.5, half_width=0.125)


def test_reports_deferral_separately_from_unreached_and_accuracy():
    result = report([call("a", "REAL", [0.1] * 15),
                     call("b", "FAKE", [0.9] * 15),
                     call("c", "REAL", [0.5] * 15),
                     call("d", "FAKE", [None] * 15)])
    for summary in result["checkpoints"]:
        assert summary["reached"] == 3
        assert summary["unreached"] == 1
        assert summary["correct_verdicts"] == 2
        assert summary["verdict_accuracy_among_reached"] == pytest.approx(2 / 3)
        assert summary["three_way"]["n_deferred"] == 1
        assert summary["three_way"]["coverage"] == pytest.approx(2 / 3)
        assert summary["three_way"]["non_deferred"]["accuracy"] == 1
        assert summary["roc_auc"] == 1


def test_no_speech_has_no_metrics_or_verdict():
    result = report([call("a", "REAL", [None] * 15)])
    for summary in result["checkpoints"]:
        assert summary["reached"] == 0
        assert summary["unreached"] == 1
        for key in ("forced_binary", "three_way", "eer", "roc_auc", "verdict_accuracy_among_reached"):
            assert summary[key] is None


def test_checkpoint_membership_changes_without_dropping_calls():
    result = report([call("a", "REAL", [0.1] * 4 + [None] * 11)])
    assert [row["reached"] for row in result["checkpoints"]] == [1, 0, 0]
    assert result["checkpoints"][0]["roc_auc"] is None
    assert len(result["calls"]) == 1


@pytest.mark.parametrize("calls", [[], [call("a", "bad", [0.1] * 15)],
    [call("a", "REAL", [0.1] * 15)] * 2, [call("", "REAL", [0.1] * 15)],
    [call("a", "REAL", [0.1] * 14)], [call("a", "REAL", [0.1] * 14 + [float("nan")])]])
def test_rejects_ambiguous_calls(calls):
    with pytest.raises(ValueError):
        report(calls)


def test_strict_json_preserves_undefined_rates_as_null():
    payload = json.loads(report_json(report([call("a", "REAL", [0.5] * 15)])))
    assert payload["checkpoints"][0]["three_way"]["non_deferred"]["accuracy"] is None
    assert payload["checkpoints"][0]["verdict_accuracy_among_reached"] == 0


@pytest.fixture
def bound_plan(tmp_path):
    manifest = Path(__file__).resolve().parents[2] / "data/manifests/for2sec_frozen_manifest.csv"
    target = tmp_path / "plan.json"
    digest = export_call_plan(manifest, target, code_commit="a" * 40)
    plan = json.loads(target.read_text())
    scores = {path: (0.1 if label == "REAL" else 0.9)
              for label, group in plan["plans"].items() for call in group["calls"] for path in call}
    return target, manifest, digest, scores


def test_bound_report_verifies_membership_and_records_digest(bound_plan):
    path, manifest, digest, scores = bound_plan
    result = report_from_plan(path, manifest, scores, expected_plan_sha256=digest,
                              threshold=0.5, half_width=0.125)
    assert result["call_plan_sha256"] == digest
    assert all(row["verdict_accuracy_among_reached"] == 1 for row in result["checkpoints"])


@pytest.mark.parametrize("change", ["missing", "extra", "wrong_hash", "membership"])
def test_bound_report_rejects_corruption(bound_plan, change):
    path, manifest, digest, scores = bound_plan
    if change == "missing":
        scores.pop(next(iter(scores)))
    elif change == "extra":
        scores["unknown"] = 0.5
    elif change == "wrong_hash":
        digest = "0" * 64
    else:
        plan = json.loads(path.read_text())
        plan["plans"]["REAL"]["calls"][0].reverse()
        path.write_text(json.dumps(plan))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        report_from_plan(path, manifest, scores, expected_plan_sha256=digest,
                         threshold=0.5, half_width=0.125)
