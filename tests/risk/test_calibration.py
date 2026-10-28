"""Platt calibration tests (Stage 10, step R1) on small synthetic arrays (no dataset access).

Convention: label 1 = spoof (positive), 0 = bonafide; higher score = more spoof;
p = 1 / (1 + exp(-(a*s + b))) = calibrated P(spoof).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.metrics import eer, roc_auc
from src.risk import calibration
from src.risk.calibration import (
    PlattCalibration,
    _fit_newton,
    _platt_targets,
    _sigmoid,
    fit_platt,
)


def test_smoothed_targets():
    # N_spoof = N_bonafide = 2: t_spoof = (2 + 1) / (2 + 2) = 3/4, t_bonafide = 1 / (2 + 2) = 1/4.
    assert _platt_targets(2, 2) == (0.75, 0.25)


def test_parameter_recovery():
    # Labels drawn from the model itself, P(spoof | s) = sigmoid(2s - 1), so the fit should recover
    # a = 2, b = -1. With 5,000 points the standard errors are about 0.05, so +/- 0.15 is ~3 SE;
    # the smoothed targets shift the optimum by O(1/N), far below the tolerance.
    rng = np.random.default_rng(2026)
    s = rng.normal(size=5000)
    y = (rng.random(5000) < _sigmoid(2.0 * s - 1.0)).astype(int)
    cal = fit_platt(y, s)
    assert cal.a == pytest.approx(2.0, abs=0.15)
    assert cal.b == pytest.approx(-1.0, abs=0.15)
    assert cal.n_spoof + cal.n_bonafide == 5000
    assert cal.n_spoof == int(y.sum())


def test_separable_data_gives_finite_parameters():
    # Plain logistic regression has no finite solution here: any a > 0 with threshold -b/a in
    # (0.3, 0.7) separates the classes, and the likelihood keeps increasing as a -> inf.
    # Targets 4/5 and 1/5 (N = 3 per class) cap the optimum, so a and b stay finite and every p
    # stays strictly inside (0, 1).
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    cal = fit_platt(labels, scores)
    assert np.isfinite(cal.a) and cal.a > 0
    assert np.isfinite(cal.b)
    p = cal.transform(scores)
    assert np.all((p > 0) & (p < 1))
    # Symmetric data around 0.5 -> p(0.5) = 0.5, i.e. b = -a/2.
    assert cal.b == pytest.approx(-cal.a / 2, abs=1e-9)


def test_calibration_is_monotonic_and_keeps_score_metrics():
    # a > 0, so p is strictly increasing in s; ROC-AUC and EER depend only on the ranking, so they
    # are identical for p and for s (the EER threshold moves, the EER rate does not).
    rng = np.random.default_rng(2026)
    s = rng.normal(size=200)
    y = (rng.random(200) < _sigmoid(2.0 * s)).astype(int)
    cal = fit_platt(y, s)
    order = np.argsort(s)
    p = cal.transform(s)
    assert np.all(np.diff(p[order]) > 0)
    assert roc_auc(y, p) == pytest.approx(roc_auc(y, s), abs=1e-12)
    assert eer(y, p)[0] == pytest.approx(eer(y, s)[0], abs=1e-12)


def test_reversed_scores_raise():
    # Scores decrease with spoof: the fit gives a < 0, which would invert the ranking.
    with pytest.raises(ValueError, match="a = .* <= 0"):
        fit_platt([0, 0, 0, 1, 1, 1], [0.9, 0.8, 0.7, 0.3, 0.2, 0.1])


@pytest.mark.filterwarnings("error")
def test_transform_extreme_scores_without_warnings():
    # a*s + b = -1e6, 0, 1e6: a naive exp(-z) would overflow at z = -1e6. The stable sigmoid only
    # takes exp of non-positive numbers, so the results are 0 (underflow), 0.5 and 1.
    cal = PlattCalibration(a=1.0, b=0.0, n_spoof=1, n_bonafide=1)
    p = cal.transform([-1e6, 0.0, 1e6])
    assert np.all((p >= 0) & (p <= 1))
    assert p.tolist() == [0.0, 0.5, 1.0]


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        ([1, 1, 1], [0.1, 0.2, 0.3]),          # one class
        ([0, 1, 1], [0.1, float("nan"), 0.3]),  # NaN score
        ([0, 1, 1], [0.1, 0.2]),                # length mismatch
        ([], []),                               # empty
    ],
)
def test_invalid_input_raises(labels, scores):
    with pytest.raises(ValueError):
        fit_platt(labels, scores)


# --- Stop rule (D44): synthetic val-B-shaped logits, numpy only -------------------------------

_KINDS = ("gamma_sep", "narrow_gap", "one_overlap")
_SCALES = (0.01, 1.0, 1000.0)


def _val_b_like(rng, kind):
    n0, n1 = 660, 669            # val-B class counts (Stage 4 split report)
    if kind == "gamma_sep":      # separable, heavy-tailed logits
        b = -rng.gamma(2.0, 4.0, n0) - 0.5
        s = rng.gamma(2.0, 4.0, n1) + 0.5
    elif kind == "narrow_gap":   # separable, tiny margin
        b = -np.abs(rng.normal(6, 2, n0)) - 1e-3
        s = np.abs(rng.normal(6, 2, n1)) + 1e-3
    elif kind == "one_overlap":  # one spoof below every bonafide (EER of about 1/669)
        b = -np.abs(rng.normal(6, 2, n0)) - 0.5
        s = np.abs(rng.normal(6, 2, n1)) + 0.5
        s[0] = -3.0
    return np.r_[b, s], np.r_[np.zeros(n0, int), np.ones(n1, int)]


def _val_b_datasets():
    # 5 datasets per kind, drawn in sequence (kinds in _KINDS order) from one rng(2026).
    rng = np.random.default_rng(2026)
    return [(kind, *_val_b_like(rng, kind)) for kind in _KINDS for _ in range(5)]


def _first_per_kind():
    # The first dataset of each kind in the _val_b_datasets() sequence.
    data = _val_b_datasets()
    return [data[5 * k] for k in range(len(_KINDS))]


def _targets_and_start(y):
    # The same smoothed targets and start point that fit_platt uses.
    n_spoof = int(y.sum())
    n_bonafide = int(len(y) - n_spoof)
    t_spoof, t_bonafide = _platt_targets(n_spoof, n_bonafide)
    t = np.where(y == 1, t_spoof, t_bonafide)
    return t, -math.log((n_bonafide + 1) / (n_spoof + 1))


def test_converges_on_val_b_like_separable_logits():
    # Newton converges quadratically near the optimum, so a converged fit needs few iterations;
    # the reviewer pre-check needed at most 11 on such data. 15 leaves margin, and any scale must
    # work, because the stop rule must not depend on the units of the score.
    for kind, s, y in _val_b_datasets():
        t, b0 = _targets_and_start(y)
        for c in _SCALES:
            cal = fit_platt(y, c * s)
            assert np.isfinite(cal.a) and cal.a > 0, (kind, c)
            assert np.isfinite(cal.b), (kind, c)
            _, _, iterations = _fit_newton(c * s, t, b0)
            assert iterations <= 15, (kind, c, iterations)


def test_fit_is_invariant_to_score_scale():
    # sigmoid(a * (c*s) + b) with a -> a/c is the same function of s, so the optimum in p does not
    # depend on c; a stop rule that does not depend on c gives the same p to float64 precision.
    for kind, s, y in _first_per_kind():
        p = [fit_platt(y, c * s).transform(c * s) for c in _SCALES]
        for other in p[1:]:
            assert np.max(np.abs(other - p[0])) <= 1e-9, kind


def test_fitted_point_is_a_minimum():
    # The convergence test must stop at the optimum, not merely stop: the loss at the fitted
    # (a, b) is no larger than at its 8 neighbours a*(1 +/- 1e-6), b +/- 1e-6 * max(1, |b|).
    for kind, s, y in _first_per_kind():
        t, _ = _targets_and_start(y)
        cal = fit_platt(y, s)

        def loss(a, b):
            z = a * s + b
            return float(np.sum(np.logaddexp(0.0, z) - t * z))

        da, db = 1e-6 * cal.a, 1e-6 * max(1.0, abs(cal.b))
        f0 = loss(cal.a, cal.b)
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if i or j:
                    assert f0 <= loss(cal.a + i * da, cal.b + j * db), (kind, i, j)


def test_stalled_iteration_fails_loudly(monkeypatch):
    # With the decrement rule disabled only the absolute gradient rule remains, and on this data
    # the gradient stalls above 1e-10 while accepted steps no longer change (a, b). The guard must
    # raise "stalled" instead of spinning to MAX_ITERATIONS.
    monkeypatch.setattr(calibration, "DECREMENT_TOLERANCE", 0.0)
    _, s, y = _first_per_kind()[0]
    with pytest.raises(ValueError, match="stalled"):
        fit_platt(y, s)
