"""Fit the MFCC/SVM baseline (D7, D9, D29, D37): Training & Tuning, research tier (step B2).

Uses eer() from the evaluation harness and scikit-learn, so it lives in the research tier (SDD
Fig. 1); src/models/baseline.py only stores and applies the fitted model and never imports this
module.

Conventions (Protocol v1.0 §4, §5; D9, D29, D37):
- features: n x 40 finite floats (MFCC mean + std); labels: 1 = spoof, 0 = bonafide, both present.
- scaler: fit_scaler on train only, applied unchanged to val-A and val-B (D37).
- selection: each (C, gamma) of the configs/baseline.yaml grid, in grid order, is trained on the
  standardized train split; the lowest val-A EER of SVC.decision_function wins; ties go to the
  smaller C, then the smaller gamma (sort key (eer, C, gamma)). val-A is used here only.
- threshold: the val-B EER point of the fitted model's score, BaselineModel.score (D29, score
  scale, forced binary). val-B is used here only.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np
import yaml
from sklearn.svm import SVC

from src.evaluation.metrics import eer
from src.models.baseline import N_FEATURES, BaselineModel, fit_scaler, from_sklearn
from src.risk.decisions import _validate_labels

# The only values the code implements; anything else in the config is refused.
_FIXED = {
    "decision": "D37",
    "features": "mfcc_mean_std_40",
    "scaling": "standardize_train_only",
    "selection": "lowest_val_a_eer",
    "tie_break": ["C", "gamma"],
    "score": "decision_function",
    "threshold": "d29_val_b_eer",
}
_CONFIG_KEYS = set(_FIXED) | {"grid"}
_GRID_KEYS = {"C", "gamma"}


@dataclass(frozen=True)
class BaselineConfig:
    """configs/baseline.yaml, validated. Grid values are floats, strictly increasing."""

    decision: str
    features: str
    scaling: str
    C: tuple[float, ...]
    gamma: tuple[float, ...]
    selection: str
    tie_break: tuple[str, ...]
    score: str
    threshold: str

    def grid(self) -> list[tuple[float, float]]:
        """The (C, gamma) pairs in grid order: C outer, gamma inner."""
        return [(c, g) for c in self.C for g in self.gamma]


def _check_keys(where: str, keys, expected: set[str]) -> None:
    missing = sorted(expected - set(keys))
    unknown = sorted(set(keys) - expected)
    if missing or unknown:
        raise ValueError(f"{where}: missing keys {missing}, unknown keys {unknown}")


def _grid_values(name: str, values) -> tuple[float, ...]:
    """A non-empty, strictly increasing tuple of positive finite numbers, or ValueError."""
    if not isinstance(values, list) or not values:
        raise ValueError(f"grid.{name} must be a non-empty list, got {values!r}")
    for v in values:
        if isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"grid.{name} values must be finite numbers > 0, got {v!r}")
    out = tuple(float(v) for v in values)
    if any(a >= b for a, b in zip(out, out[1:])):
        raise ValueError(f"grid.{name} must be strictly increasing, got {values!r}")
    return out


def load_baseline_config(path) -> BaselineConfig:
    """Read configs/baseline.yaml with yaml.safe_load; strict keys; raise ValueError if invalid."""
    path = Path(path)
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read baseline config {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must hold a YAML mapping")
    _check_keys(str(path), cfg.keys(), _CONFIG_KEYS)
    for key, expected in _FIXED.items():
        if cfg[key] != expected:
            raise ValueError(f"{key} must be {expected!r} (D37), got {cfg[key]!r}")
    grid = cfg["grid"]
    if not isinstance(grid, dict):
        raise ValueError(f"grid must be a mapping, got {grid!r}")
    _check_keys("grid", grid.keys(), _GRID_KEYS)
    return BaselineConfig(
        decision=cfg["decision"],
        features=cfg["features"],
        scaling=cfg["scaling"],
        C=_grid_values("C", grid["C"]),
        gamma=_grid_values("gamma", grid["gamma"]),
        selection=cfg["selection"],
        tie_break=tuple(cfg["tie_break"]),
        score=cfg["score"],
        threshold=cfg["threshold"],
    )


@dataclass(frozen=True)
class GridRow:
    """One grid configuration and its val-A result."""

    C: float
    gamma: float
    val_a_eer: float
    n_support: int


def _to_json(report) -> str:
    return json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True, allow_nan=False) + "\n"


@dataclass(frozen=True)
class SelectionReport:
    """All grid rows in grid order, the selected (C, gamma) and the split sizes used."""

    rows: tuple[GridRow, ...]
    selected_C: float
    selected_gamma: float
    n_train: int
    n_val_a: int

    def to_json(self) -> str:
        return _to_json(self)


@dataclass(frozen=True)
class ThresholdReport:
    """The D29 threshold on the score scale, the val-B EER there and the val-B class counts."""

    threshold: float
    val_b_eer: float
    n_val_b: int
    n_spoof_val_b: int
    n_bonafide_val_b: int

    def to_json(self) -> str:
        return _to_json(self)


def _split(features, labels, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (n x 40 finite floats, 0/1 int labels with both classes), or raise ValueError."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != N_FEATURES or x.shape[0] == 0:
        raise ValueError(f"{name} features must have shape (n, {N_FEATURES}), got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} features contain {int((~np.isfinite(x)).sum())} NaN or "
                         "infinite values")
    try:
        y = _validate_labels(labels)
    except ValueError as exc:
        raise ValueError(f"{name} labels: {exc}") from exc
    if len(y) != len(x):
        raise ValueError(f"{name}: {len(x)} feature rows but {len(y)} labels")
    return x, y


def select_baseline(train_X, train_y, val_a_X, val_a_y, *, config: BaselineConfig):
    """Train every grid configuration on train and pick the lowest val-A EER (D37, D9).

    Returns (svc, (mean, std), SelectionReport). The scaler is fitted on train only. Ties in val-A
    EER go to the smaller C, then the smaller gamma: the minimum of the sort key (eer, C, gamma).
    """
    if not isinstance(config, BaselineConfig):
        raise ValueError(f"config must be a BaselineConfig, got {type(config).__name__}")
    x_train, y_train = _split(train_X, train_y, "train")
    x_val_a, y_val_a = _split(val_a_X, val_a_y, "val-A")
    mean, std = fit_scaler(x_train)
    z_train = (x_train - mean) / std
    z_val_a = (x_val_a - mean) / std

    rows, models = [], {}
    for C, gamma in config.grid():
        svc = SVC(kernel="rbf", C=C, gamma=gamma).fit(z_train, y_train)
        rate, _ = eer(y_val_a, svc.decision_function(z_val_a))
        rows.append(GridRow(C=C, gamma=gamma, val_a_eer=rate, n_support=int(svc.n_support_.sum())))
        models[(C, gamma)] = svc
    best = min(rows, key=lambda r: (r.val_a_eer, r.C, r.gamma))
    report = SelectionReport(
        rows=tuple(rows),
        selected_C=best.C,
        selected_gamma=best.gamma,
        n_train=len(y_train),
        n_val_a=len(y_val_a),
    )
    return models[(best.C, best.gamma)], (mean, std), report


def _val_b_eer_point(scores, labels) -> tuple[float, float]:
    return eer(labels, scores)


def fit_baseline_threshold(model_scores_val_b, val_b_y) -> float:
    """The D29 threshold: the val-B EER point of the baseline score (score scale, forced binary)."""
    return _val_b_eer_point(model_scores_val_b, val_b_y)[1]


def fit_baseline(train_X, train_y, val_a_X, val_a_y, val_b_X, val_b_y, *, config: BaselineConfig,
                 version: str):
    """Select on val-A, then threshold on val-B; return (model, SelectionReport, ThresholdReport).

    The val-B scores come from BaselineModel.score (the NumPy scorer that Stage 13 uses), so the
    threshold and every stored score share one scale.
    """
    x_val_b, y_val_b = _split(val_b_X, val_b_y, "val-B")
    svc, (mean, std), selection = select_baseline(train_X, train_y, val_a_X, val_a_y,
                                                  config=config)
    # Threshold 0.0 is a placeholder until the val-B threshold is known; it does not affect score.
    unthresholded = from_sklearn(svc, mean=mean, std=std, threshold=0.0, version=version)
    scores = unthresholded.score(x_val_b)
    rate, threshold = _val_b_eer_point(scores, y_val_b)
    model = dataclasses.replace(unthresholded, threshold=threshold)
    report = ThresholdReport(
        threshold=threshold,
        val_b_eer=rate,
        n_val_b=len(y_val_b),
        n_spoof_val_b=int(y_val_b.sum()),
        n_bonafide_val_b=int(len(y_val_b) - y_val_b.sum()),
    )
    return model, selection, report
