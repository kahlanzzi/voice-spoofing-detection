"""Platt calibration of detector scores (Protocol v1.0 §6, FR-18): p = 1 / (1 + exp(-(a*s + b))).

Fitted on val-B only (D2, D9) by Platt's method with smoothed targets (Platt 1999; Lin, Lin and
Weng 2007), using Newton's method with a backtracking line search in NumPy. The fitted slope a
must be > 0, so the calibration is strictly increasing and never changes the ranking of scores.

Stop rule (D44): stop when the gradient norm is < GRADIENT_TOLERANCE or when the Newton decrement
lambda^2/2 = -g.d/2 (about f - f*, independent of the score scale; Boyd and Vandenberghe 2004,
§9.5.1) is <= DECREMENT_TOLERANCE. A Newton direction that is not a descent direction
(lambda^2/2 < -DECREMENT_TOLERANCE) raises ValueError, so the decrement stop never accepts a
non-optimal point; an accepted line-search step that leaves (a, b) unchanged raises ValueError
("stalled") instead of running on to MAX_ITERATIONS.

Conventions: labels 1 = spoof, 0 = bonafide; higher score = more likely spoof; p = P(spoof).
Part of the shared core: must not import src.evaluation (SDD Fig. 1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.risk.decisions import _require_finite_number, _validate, _validate_scores

MAX_ITERATIONS = 100
GRADIENT_TOLERANCE = 1e-10
DECREMENT_TOLERANCE = 1e-12  # Newton decrement stop, lambda^2/2 <= this (D44)
_ARMIJO = 1e-4          # sufficient-decrease constant of the line search
_MIN_STEP = 1e-10       # smallest line-search step before giving up
_HESSIAN_RIDGE = 1e-12  # added to the Hessian diagonal, as in Lin, Lin and Weng (2007)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """1 / (1 + exp(-z)) without overflow: exp is only taken of non-positive numbers."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


@dataclass(frozen=True)
class PlattCalibration:
    """Fitted Platt parameters and the class counts of the fit data (val-B)."""

    a: float
    b: float
    n_spoof: int
    n_bonafide: int

    def __post_init__(self) -> None:
        _require_finite_number("a", self.a)
        _require_finite_number("b", self.b)
        if self.a <= 0:
            raise ValueError(f"a must be > 0 so calibration keeps the ranking, got {self.a!r}")
        for name in ("n_spoof", "n_bonafide"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")

    def transform(self, scores) -> np.ndarray:
        """Calibrated spoof probabilities p = 1 / (1 + exp(-(a*s + b))), in [0, 1]."""
        s = _validate_scores(scores)
        return _sigmoid(self.a * s + self.b)


def _platt_targets(n_spoof: int, n_bonafide: int) -> tuple[float, float]:
    """Smoothed targets (t_spoof, t_bonafide) = ((N+ + 1) / (N+ + 2), 1 / (N- + 2)) (Platt 1999)."""
    return (n_spoof + 1) / (n_spoof + 2), 1.0 / (n_bonafide + 2)


def _fit_newton(s: np.ndarray, t: np.ndarray, b0: float) -> tuple[float, float, int]:
    """Minimise the cross-entropy of sigmoid(a*s + b) against targets t; return (a, b, iterations)."""

    def loss(a: float, b: float) -> float:
        z = a * s + b
        # -(t log p + (1 - t) log(1 - p)) = log(1 + exp(z)) - t z, computed stably.
        return float(np.sum(np.logaddexp(0.0, z) - t * z))

    a, b = 0.0, b0
    f = loss(a, b)
    for iteration in range(MAX_ITERATIONS + 1):
        p = _sigmoid(a * s + b)
        r = p - t
        grad = np.array([np.dot(r, s), np.sum(r)])
        if np.linalg.norm(grad) < GRADIENT_TOLERANCE:
            return a, b, iteration
        w = p * (1.0 - p)
        hess = np.array([[np.dot(w, s * s), np.dot(w, s)], [np.dot(w, s), np.sum(w)]])
        hess += _HESSIAN_RIDGE * np.eye(2)
        direction = -np.linalg.solve(hess, grad)
        slope = float(grad @ direction)  # <= 0 when the Hessian is positive definite; guarded below
        half_decrement = -slope / 2  # lambda^2/2; >= 0 for a descent direction
        if half_decrement < -DECREMENT_TOLERANCE:
            raise ValueError(
                f"Platt fit: Newton direction is not a descent direction at iteration {iteration} "
                f"(Newton decrement^2/2 {half_decrement:.3g})"
            )
        if half_decrement <= DECREMENT_TOLERANCE:
            return a, b, iteration
        if iteration == MAX_ITERATIONS:
            break

        step = 1.0
        while step >= _MIN_STEP:
            a_new, b_new = a + step * direction[0], b + step * direction[1]
            f_new = loss(a_new, b_new)
            if f_new <= f + _ARMIJO * step * slope:
                break
            step /= 2.0
        else:
            raise ValueError(
                f"Platt fit: line search failed at iteration {iteration} "
                f"(gradient norm {np.linalg.norm(grad):.3g})"
            )
        if float(a_new) == a and float(b_new) == b:
            raise ValueError(
                f"Platt fit: stalled at iteration {iteration} (gradient norm "
                f"{np.linalg.norm(grad):.3g}, Newton decrement^2/2 {half_decrement:.3g}); "
                "the step no longer changes a or b"
            )
        a, b, f = float(a_new), float(b_new), f_new

    raise ValueError(
        f"Platt fit did not converge in {MAX_ITERATIONS} iterations "
        f"(gradient norm {np.linalg.norm(grad):.3g})"
    )


def fit_platt(labels, scores) -> PlattCalibration:
    """Fit Platt calibration on labelled scores (val-B only, D2/D9); both classes are required.

    Starts from a = 0, b = log((N_spoof + 1) / (N_bonafide + 1)), i.e. p = smoothed prior.
    Raises ValueError on invalid input, on non-convergence, or if the fitted a <= 0 (scores do not
    increase with spoof on the fit data).
    """
    y, s = _validate(labels, scores)
    n_spoof = int(y.sum())
    n_bonafide = int(len(y) - n_spoof)
    t_spoof, t_bonafide = _platt_targets(n_spoof, n_bonafide)
    t = np.where(y == 1, t_spoof, t_bonafide)

    a, b, _ = _fit_newton(s, t, b0=-math.log((n_bonafide + 1) / (n_spoof + 1)))
    if a <= 0:
        raise ValueError(
            f"Platt fit gave a = {a:.6g} <= 0: scores do not increase with spoof on the fit data, "
            "and calibration must not change the ranking"
        )
    return PlattCalibration(a=a, b=b, n_spoof=n_spoof, n_bonafide=n_bonafide)
