import pytest

from src.evaluation.live_warmup_report import checkpoint_report


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
