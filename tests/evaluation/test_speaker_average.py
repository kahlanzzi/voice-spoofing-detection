"""Synthetic-only tests of speaker weighting and explicit undefined handling."""
from dataclasses import replace
import math

import pytest

from src.evaluation.speaker_average import speaker_average
from src.evaluation.speakers import per_speaker_metrics


def test_equal_speaker_weight_despite_unequal_window_counts():
    # A: 100 correct windows; B: two reversed windows. Macro=.5, pooled=100/102.
    rows = per_speaker_metrics([0, 1] * 50 + [0, 1],
        [0.1, 0.9] * 50 + [0.9, 0.1], ["A"] * 100 + ["B"] * 2,
        threshold=0.5, half_width=0)
    result = speaker_average(rows, undefined_policy="raise")
    assert len(result) == 18
    for name in ("binary.accuracy", "roc_auc", "eer", "non_deferred.f1"):
        assert result[name].value == 0.5
        assert result[name].n_speakers == result[name].n_defined == 2
        assert result[name].undefined_speakers == ()
    assert result["coverage"].value == 1
    assert result["deferral_rate"].value == 0
    assert "n" not in result and "confusion" not in result
    assert speaker_average(reversed(rows), undefined_policy="raise") == result


def test_undefined_speaker_is_identified_and_not_silently_excluded():
    rows = per_speaker_metrics([0, 1, 0], [0.1, 0.9, 0.2], ["A", "A", "B"],
        threshold=0.5, half_width=0)
    result = speaker_average(rows, undefined_policy="propagate")
    assert math.isnan(result["eer"].value)
    assert result["eer"].undefined_speakers == ("B",)
    assert result["eer"].n_defined == 1
    assert result["eer"].n_speakers == 2
    assert result["binary.accuracy"].value == 1
    with pytest.raises(ValueError, match="undefined.*B"):
        speaker_average(rows, undefined_policy="raise")


def test_all_deferred_retains_coverage_but_marks_non_deferred_undefined():
    rows = per_speaker_metrics([0, 1, 0, 1], [0.5] * 4, ["B", "B", "A", "A"],
        threshold=0.5, half_width=0.1)
    result = speaker_average(rows, undefined_policy="propagate")
    assert result["coverage"].value == 0
    assert result["deferral_rate"].value == 1
    assert math.isnan(result["non_deferred.accuracy"].value)
    assert result["non_deferred.accuracy"].n_defined == 0
    assert result["non_deferred.accuracy"].undefined_speakers == ("A", "B")


def test_one_speaker_and_required_policy():
    rows = per_speaker_metrics([0, 1], [0.1, 0.9], ["A", "A"],
        threshold=0.5, half_width=0)
    assert speaker_average(rows, undefined_policy="raise")["binary.accuracy"].value == 1
    with pytest.raises(TypeError):
        speaker_average(rows)
    with pytest.raises(ValueError, match="duplicate"):
        speaker_average(rows + rows, undefined_policy="propagate")


@pytest.mark.parametrize("rows", [[], None, [42], ["speaker"]])
def test_invalid_rows_raise(rows):
    with pytest.raises(ValueError):
        speaker_average(rows, undefined_policy="propagate")


@pytest.mark.parametrize("policy", [None, "omit", "", 1])
def test_unknown_policy_raises(policy):
    with pytest.raises(ValueError, match="undefined_policy"):
        speaker_average([], undefined_policy=policy)


@pytest.mark.parametrize("value", [float("inf"), -0.1, 1.1, True, "0.5"])
def test_invalid_metric_rejected(value):
    row, = per_speaker_metrics([0, 1], [0.1, 0.9], ["A", "A"],
        threshold=0.5, half_width=0)
    with pytest.raises(ValueError, match="rates"):
        speaker_average([replace(row, eer=value)], undefined_policy="propagate")


@pytest.mark.parametrize("speaker", ["", " A", None])
def test_invalid_speaker_id_rejected(speaker):
    row, = per_speaker_metrics([0, 1], [0.1, 0.9], ["A", "A"],
        threshold=0.5, half_width=0)
    with pytest.raises(ValueError, match="speaker IDs"):
        speaker_average([replace(row, source_speaker=speaker)], undefined_policy="propagate")
