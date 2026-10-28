"""D44 pre-check: Platt stop rule on synthetic val-B-shaped logits (no dataset is read).

Reproduces the evidence cited in the D44 register row (docs/decisions/README.md):
  - how often the e2374cc stop rule (absolute gradient norm < 1e-10 only) raises on fits that
    have converged, per distribution and per score scale;
  - that the D44 rule in src/risk/calibration.py converges on the same fits, and in how many
    Newton iterations;
  - the float64 noise floor of the Newton decrement lambda^2/2, which the D44 bound (1e-12)
    must sit above.

Synthetic only: numpy draws with seed 2026, 660 bonafide / 669 spoof (val-B class counts, Stage 4
split report). No audio, cache, checkpoint, val-B or test data. numpy only; no new dependency.

Run from the repository root:
    python docs/audit_evidence/stage10_d44/platt_stop_precheck.py > docs/audit_evidence/stage10_d44/platt_stop_precheck.txt
"""
from __future__ import annotations

import math
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root, for `src`

from src.risk import calibration  # noqa: E402  (the D44 rule under test)
from src.risk.calibration import _fit_newton, _platt_targets, _sigmoid  # noqa: E402

SEED = 2026
N_BONAFIDE, N_SPOOF = 660, 669
KINDS = ("normal_sep", "gamma_sep", "gamma_wide", "narrow_gap", "one_overlap", "asym",
         "overlap_5pct")
DATASETS_PER_KIND = 30
SCALES = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
FLOOR_ITERATIONS = 30  # unstopped Newton iterations used to measure the noise floor
FLOOR_FROM = 15        # iterations >= this are past convergence on every dataset here


def _fit_newton_e2374cc(s, t, b0):
    """The stop rule on main at e2374cc (before D44), copied verbatim except the error text.

    Absolute gradient norm < 1e-10 only; ValueError after 100 iterations or a failed line search.
    """
    def loss(a, b):
        z = a * s + b
        return float(np.sum(np.logaddexp(0.0, z) - t * z))

    a, b = 0.0, b0
    f = loss(a, b)
    for iteration in range(100 + 1):
        p = _sigmoid(a * s + b)
        r = p - t
        grad = np.array([np.dot(r, s), np.sum(r)])
        if np.linalg.norm(grad) < 1e-10:
            return a, b, iteration
        if iteration == 100:
            break
        w = p * (1.0 - p)
        hess = np.array([[np.dot(w, s * s), np.dot(w, s)], [np.dot(w, s), np.sum(w)]])
        hess += 1e-12 * np.eye(2)
        direction = -np.linalg.solve(hess, grad)
        slope = float(grad @ direction)
        step = 1.0
        while step >= 1e-10:
            a_new, b_new = a + step * direction[0], b + step * direction[1]
            f_new = loss(a_new, b_new)
            if f_new <= f + 1e-4 * step * slope:
                break
            step /= 2.0
        else:
            raise ValueError("line search failed")
        a, b, f = float(a_new), float(b_new), f_new
    raise ValueError("did not converge in 100 iterations")


def _decrement_path(s, t, b0):
    """lambda^2/2 at each of FLOOR_ITERATIONS Newton iterations, with no stop rule at all."""
    def loss(a, b):
        z = a * s + b
        return float(np.sum(np.logaddexp(0.0, z) - t * z))

    a, b = 0.0, b0
    f = loss(a, b)
    path = []
    for _ in range(FLOOR_ITERATIONS):
        p = _sigmoid(a * s + b)
        r = p - t
        grad = np.array([np.dot(r, s), np.sum(r)])
        w = p * (1.0 - p)
        hess = np.array([[np.dot(w, s * s), np.dot(w, s)], [np.dot(w, s), np.sum(w)]])
        hess += 1e-12 * np.eye(2)
        direction = -np.linalg.solve(hess, grad)
        slope = float(grad @ direction)
        path.append(-slope / 2)
        step = 1.0
        while step >= 1e-10:
            a_new, b_new = a + step * direction[0], b + step * direction[1]
            f_new = loss(a_new, b_new)
            if f_new <= f + 1e-4 * step * slope:
                break
            step /= 2.0
        a, b, f = float(a_new), float(b_new), f_new
    return np.array(path)


def _dataset(rng, kind):
    """Synthetic CNN-like logits: bonafide negative, spoof positive (higher = more likely spoof)."""
    n0, n1 = N_BONAFIDE, N_SPOOF
    if kind == "normal_sep":      # separable, light tails
        b, s = -np.abs(rng.normal(6, 2, n0)) - 0.5, np.abs(rng.normal(6, 2, n1)) + 0.5
    elif kind == "gamma_sep":     # separable, heavy tails
        b, s = -rng.gamma(2, 4, n0) - 0.5, rng.gamma(2, 4, n1) + 0.5
    elif kind == "gamma_wide":    # separable, very wide logits
        b, s = -rng.gamma(1, 10, n0) - 0.5, rng.gamma(1, 10, n1) + 0.5
    elif kind == "narrow_gap":    # separable, tiny margin
        b, s = -np.abs(rng.normal(6, 2, n0)) - 1e-3, np.abs(rng.normal(6, 2, n1)) + 1e-3
    elif kind == "one_overlap":   # one spoof below every bonafide (EER of about 1/669)
        b, s = -np.abs(rng.normal(6, 2, n0)) - 0.5, np.abs(rng.normal(6, 2, n1)) + 0.5
        s[0] = -3.0
    elif kind == "asym":          # spoof logits much larger than bonafide
        b, s = -rng.gamma(2, 2, n0) - 0.5, rng.gamma(3, 8, n1) + 0.5
    elif kind == "overlap_5pct":  # overlapping classes (not separable)
        b, s = rng.normal(-4, 2, n0), rng.normal(4, 2, n1)
    else:
        raise ValueError(kind)
    return np.r_[b, s], np.r_[np.zeros(n0, int), np.ones(n1, int)]


def main() -> None:
    rng = np.random.default_rng(SEED)
    old_fail = {(k, c): 0 for k in KINDS for c in SCALES}
    new_errors = []
    new_max_iterations = 0
    floors = []
    n_fits = 0
    for kind in KINDS:
        for index in range(DATASETS_PER_KIND):
            s0, y = _dataset(rng, kind)
            n_spoof = int(y.sum())
            t_spoof, t_bonafide = _platt_targets(n_spoof, len(y) - n_spoof)
            t = np.where(y == 1, t_spoof, t_bonafide)
            b0 = -math.log((len(y) - n_spoof + 1) / (n_spoof + 1))  # fit_platt's start point
            for c in SCALES:
                s = c * s0
                n_fits += 1
                try:
                    _fit_newton_e2374cc(s, t, b0)
                except ValueError:
                    old_fail[(kind, c)] += 1
                try:
                    _, _, iterations = _fit_newton(s, t, b0)
                    new_max_iterations = max(new_max_iterations, iterations)
                except ValueError as error:
                    new_errors.append((kind, index, c, str(error)))
                floors.append(float(_decrement_path(s, t, b0)[FLOOR_FROM:].max()))

    print("D44 pre-check: Platt stop rule on synthetic val-B-shaped logits")
    print(f"python {platform.python_version()}, numpy {np.__version__}, seed {SEED}, "
          f"{N_BONAFIDE} bonafide / {N_SPOOF} spoof, {DATASETS_PER_KIND} datasets per kind")
    print(f"D44 constants in src/risk/calibration.py: GRADIENT_TOLERANCE "
          f"{calibration.GRADIENT_TOLERANCE:g}, DECREMENT_TOLERANCE {calibration.DECREMENT_TOLERANCE:g}")
    print()
    print("e2374cc rule (gradient norm < 1e-10 only): failed fits per kind and score scale")
    print("kind".ljust(14) + "".join(f"x{c:g}".rjust(8) for c in SCALES) + "   total")
    for kind in KINDS:
        row = [old_fail[(kind, c)] for c in SCALES]
        print(kind.ljust(14) + "".join(f"{v:8d}" for v in row) + f"{sum(row):8d}")
    per_scale = [sum(old_fail[(k, c)] for k in KINDS) for c in SCALES]
    print("all".ljust(14) + "".join(f"{v:8d}" for v in per_scale) + f"{sum(per_scale):8d}")
    print(f"e2374cc rule: {sum(per_scale)} of {n_fits} fits failed")
    print()
    print(f"D44 rule (repository code): {len(new_errors)} of {n_fits} fits failed; "
          f"at most {new_max_iterations} Newton iterations")
    for kind, index, c, message in new_errors:
        print(f"  error: {kind} #{index} x{c:g}: {message}")
    floors_array = np.array(floors)
    worst = float(floors_array.max())
    print()
    print(f"Noise floor of lambda^2/2 after convergence (max over iterations {FLOOR_FROM}-"
          f"{FLOOR_ITERATIONS - 1}, no stop rule): median {np.median(floors_array):.1e}, "
          f"worst {worst:.1e}")
    print(f"DECREMENT_TOLERANCE / worst floor = {calibration.DECREMENT_TOLERANCE / worst:.0f}")


if __name__ == "__main__":
    main()
