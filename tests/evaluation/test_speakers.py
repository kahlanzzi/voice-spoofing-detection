"""Invented scores only: hand calculations and validation for speaker tables."""
import math

import numpy as np
import pytest

from src.evaluation.speakers import per_speaker_metrics


def test_hand_calculated_speaker_table():
    # A is perfectly ranked. B is completely reversed. One shared policy.
    rows = per_speaker_metrics(
        [0, 1, 0, 1], [0.9, 0.1, 0.1, 0.9], ["B", "B", "A", "A"],
        threshold=0.5, half_width=0.1,
    )
    a, b = rows
    assert (a.source_speaker, b.source_speaker) == ("A", "B")
    assert a.n == b.n == 2
    assert (a.eer, a.roc_auc, a.binary.accuracy) == (0, 1, 1)
    assert (b.eer, b.roc_auc, b.binary.accuracy) == (1, 0, 0)
    assert a.three_way.confusion == ((1, 0, 0), (0, 1, 0))
    assert b.three_way.confusion == ((0, 1, 0), (1, 0, 0))


def test_shared_frozen_threshold_is_not_replaced_by_speaker_eer_threshold():
    row, = per_speaker_metrics(
        [0, 1], [0.7, 0.8], ["A", "A"], threshold=0.5, half_width=0,
    )
    assert row.eer == 0  # Perfect ranking, but supplied threshold yields a false positive.
    assert row.binary.fp == 1
    assert row.binary.accuracy == 0.5


def test_band_boundaries_and_all_deferred_speaker():
    a, b = per_speaker_metrics(
        [0, 1, 0, 1], [0.25, 0.75, 0.4, 0.6], ["A", "A", "B", "B"],
        threshold=0.5, half_width=0.25,
    )
    assert a.three_way.confusion == ((0, 0, 1), (0, 1, 0))
    assert a.three_way.coverage == 0.5
    assert b.three_way.n_deferred == 2
    assert b.three_way.coverage == 0
    assert math.isnan(b.three_way.non_deferred.accuracy)


def test_single_class_speaker_is_retained_with_undefined_ranking_metrics():
    a, b = per_speaker_metrics(
        [0, 0, 1], [0.1, 0.2, 0.9], ["A", "A", "B"],
        threshold=0.5, half_width=0.1,
    )
    for row in (a, b):
        assert math.isnan(row.eer)
        assert math.isnan(row.roc_auc)
        assert row.binary.accuracy == 1
    assert math.isnan(a.binary.fnr)
    assert math.isnan(b.binary.fpr)


def test_row_order_does_not_change_results_or_merge_speaker_ids():
    labels = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.2, 0.8])
    speakers = np.array(["A", "A", "a", "a", "B", "B"])
    permutation = [5, 2, 0, 4, 1, 3]
    expected = per_speaker_metrics(labels, scores, speakers, threshold=0.5, half_width=0)
    actual = per_speaker_metrics(
        labels[permutation], scores[permutation], speakers[permutation],
        threshold=0.5, half_width=0,
    )
    assert actual == expected
    assert [row.source_speaker for row in actual] == ["A", "B", "a"]
    assert sum(row.n for row in actual) == len(labels)


@pytest.mark.parametrize("speakers", [[], ["A"], [["A", "B"]], ["", "B"],
                                        [None, "B"], [1, "B"], [" A", "B"],
                                        ["A", "B "], [float("nan"), "B"]])
def test_invalid_speaker_ids_raise(speakers):
    with pytest.raises(ValueError, match="source_speakers"):
        per_speaker_metrics([0, 1], [0.1, 0.9], speakers, threshold=0.5, half_width=0)


@pytest.mark.parametrize("labels,scores", [([], []), ([0], [0.1, 0.9]),
    ([0, 2], [0.1, 0.9]), ([0, 1], [float("nan"), 0.9]),
    ([0, 1], [0.1, float("inf")]), ([[0, 1]], [[0.1, 0.9]])])
def test_invalid_labels_or_scores_raise(labels, scores):
    with pytest.raises(ValueError):
        per_speaker_metrics(labels, scores, ["A", "A"], threshold=0.5, half_width=0)


@pytest.mark.parametrize("threshold,half_width", [(float("nan"), 0), (0.5, -0.1),
    (0.5, float("inf")), (True, 0), (0.5, "0.1")])
def test_invalid_policy_raises(threshold, half_width):
    with pytest.raises(ValueError):
        per_speaker_metrics([0, 1], [0.1, 0.9], ["A", "A"],
                            threshold=threshold, half_width=half_width)
