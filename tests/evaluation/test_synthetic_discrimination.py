"""Synthetic clustered EER/AUC integration checks under approved D4/D11."""
import numpy as np
import pytest

from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.metrics import eer, roc_auc


@pytest.mark.parametrize("metric", ["eer", "auc"])
def test_clustered_discrimination_intervals_match_count_oracle(metric):
    # Single-class speakers deliberately make some whole-speaker draws invalid.
    # Unequal sizes and different error rates expose row sampling or lost duplicates.
    sizes = np.array([5, 10, 15, 20, 25, 30, 35, 40])
    classes = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    errors = np.array([0, 2, 6, 10, 1, 6, 14, 20])
    labels, scores, speakers = [], [], []
    for sid, (size, label, wrong) in enumerate(zip(sizes, classes, errors)):
        labels.extend([label] * size)
        scores.extend([0.8 if label == 0 else 0.2] * wrong)
        scores.extend([0.2 if label == 0 else 0.8] * (size - wrong))
        speakers.extend([sid] * size)
    labels, scores = np.asarray(labels), np.asarray(scores)
    statistic = (lambda idx: eer(labels[idx], scores[idx])[0]) if metric == "eer" else (
        lambda idx: roc_auc(labels[idx], scores[idx]))
    result = bootstrap_ci(statistic, len(labels), clusters=speakers)

    # Independent two-score ROC oracle, using only aggregate class/error counts.
    # ROC vertices are (0,0), (FPR,TPR), (1,1); ties get half credit in AUC.
    draws = np.random.default_rng(2026).integers(0, 8, size=(2000, 8))
    draws = np.vstack([np.arange(8), draws])  # Full-data estimate, then resamples.
    negative = classes[draws] == 0
    n0 = np.where(negative, sizes[draws], 0).sum(axis=1)
    n1 = np.where(~negative, sizes[draws], 0).sum(axis=1)
    valid = (n0 > 0) & (n1 > 0)
    fp = np.where(negative, errors[draws], 0).sum(axis=1)[valid]
    fn = np.where(~negative, errors[draws], 0).sum(axis=1)[valid]
    fpr, tpr = fp / n0[valid], 1 - fn / n1[valid]
    if metric == "auc":
        expected = (1 + tpr - fpr) / 2
    else:
        expected = np.empty_like(fpr)
        first_segment = fpr + tpr >= 1
        expected[first_segment] = fpr[first_segment] / (fpr + tpr)[first_segment]
        expected[~first_segment] = (1 - tpr[~first_segment]) / (2 - tpr - fpr)[~first_segment]
    assert result.estimate == pytest.approx(expected[0])
    assert (result.low, result.high) == pytest.approx(
        np.percentile(expected[1:], [2.5, 97.5], method="linear"))
    assert result.n_invalid == int((~valid[1:]).sum())
    assert 0 < result.n_invalid < 2000
    assert (result.n_resamples, result.confidence) == (2000, 0.95)
