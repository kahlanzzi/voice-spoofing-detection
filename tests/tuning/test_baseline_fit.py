"""Baseline fitting (step B2; D9, D29, D37): synthetic 40-feature data only, seed 2026.

Convention: label 1 = spoof (positive), 0 = bonafide; higher score = more spoof.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import sklearn

import src.tuning.baseline_fit as fit_module
from src.evaluation.metrics import eer
from src.models.baseline import N_FEATURES, artifact_sha256, save
from src.risk.decisions import AUTHENTIC, SYNTHETIC
from src.tuning.baseline_fit import (
    BaselineConfig,
    fit_baseline,
    fit_baseline_threshold,
    load_baseline_config,
    select_baseline,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "baseline.yaml"
# Different sizes per split, so a spy can tell which split's labels reached eer().
N_TRAIN, N_VAL_A, N_VAL_B = 60, 40, 50  # per class


def _data(n_per_class: int, seed: int, shift: float = 0.25):
    """Two Gaussian classes in 40 dimensions; spoof (1) is shifted by +shift in every feature.

    Features have different scales (1..40), as raw MFCC statistics do, so scaling matters. The
    small shift keeps the classes overlapping, so val-A EERs differ across the grid.
    """
    rng = np.random.default_rng(seed)
    scales = np.arange(1, N_FEATURES + 1, dtype=float)
    bona = rng.normal(0.0, 1.0, (n_per_class, N_FEATURES))
    spoof = rng.normal(shift, 1.0, (n_per_class, N_FEATURES))
    x = np.vstack([bona, spoof]) * scales
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return x, y


def _splits():
    # One seed (2026) derives the three split seeds, so the splits are independent but fixed.
    seeds = [int(s) for s in np.random.default_rng(2026).integers(0, 2**31, 3)]
    return _data(N_TRAIN, seeds[0]), _data(N_VAL_A, seeds[1]), _data(N_VAL_B, seeds[2])


@pytest.fixture(scope="module")
def config() -> BaselineConfig:
    return load_baseline_config(CONFIG_PATH)


def _fit(config, val_b=None, version="baseline-test"):
    train, val_a, default_val_b = _splits()
    val_b = default_val_b if val_b is None else val_b
    return fit_baseline(*train, *val_a, *val_b, config=config, version=version)


# 1. Config drift guard --------------------------------------------------------------------------

def test_config_equals_d37(config):
    # configs/baseline.yaml must say exactly what D37 says. If this fails, the config drifted from
    # the approved decision: changing it needs a new decision, not a test edit.
    assert config == BaselineConfig(
        decision="D37",
        features="mfcc_mean_std_40",
        scaling="standardize_train_only",
        C=(0.1, 1.0, 10.0, 100.0),
        gamma=(0.001, 0.01, 0.025, 0.1),
        selection="lowest_val_a_eer",
        tie_break=("C", "gamma"),
        score="decision_function",
        threshold="d29_val_b_eer",
    )
    assert len(config.grid()) == 16


# 2. Full grid, grid order, lowest val-A EER -----------------------------------------------------

def test_all_16_configurations_in_grid_order_and_lowest_eer_selected(config):
    _, selection, _ = _fit(config)
    # The report keeps every configuration, in grid order (C outer, gamma inner).
    assert [(r.C, r.gamma) for r in selection.rows] == config.grid()
    assert (selection.n_train, selection.n_val_a) == (2 * N_TRAIN, 2 * N_VAL_A)
    # The selected configuration has the lowest val-A EER, and is the first in (eer, C, gamma).
    best = min(r.val_a_eer for r in selection.rows)
    selected = next(r for r in selection.rows
                    if (r.C, r.gamma) == (selection.selected_C, selection.selected_gamma))
    assert selected.val_a_eer == best
    assert (selected.C, selected.gamma) == min(
        (r.C, r.gamma) for r in selection.rows if r.val_a_eer == best)
    # The EERs are not all equal on this data, so the selection is a real choice.
    assert len({r.val_a_eer for r in selection.rows}) > 1


# 3. Tie rule ------------------------------------------------------------------------------------

def test_tie_goes_to_smallest_c_then_smallest_gamma(config, monkeypatch):
    # Every configuration gets the same val-A EER, so only the tie rule decides: smaller C first,
    # then smaller gamma, i.e. the first grid point (0.1, 0.001).
    monkeypatch.setattr(fit_module, "eer", lambda labels, scores: (0.25, 0.0))
    train, val_a, _ = _splits()
    svc, _, selection = select_baseline(*train, *val_a, config=config)
    assert (selection.selected_C, selection.selected_gamma) == (0.1, 0.001)
    assert (svc.C, svc.gamma) == (0.1, 0.001)


# 4. Data separation (D9) ------------------------------------------------------------------------

def test_val_a_only_for_selection_val_b_only_for_threshold(config, monkeypatch):
    calls = []

    def spy(labels, scores):
        calls.append(len(labels))
        return eer(labels, scores)

    monkeypatch.setattr(fit_module, "eer", spy)
    model, selection, _ = _fit(config)
    # 16 selection calls see only val-A labels, then the threshold call sees only val-B labels.
    assert calls == [2 * N_VAL_A] * 16 + [2 * N_VAL_B]

    # A different val-B changes the threshold but not the selection: val-B plays no part in it.
    other_val_b = _data(N_VAL_B, seed=7, shift=0.5)
    model2, selection2, _ = _fit(config, val_b=other_val_b)
    assert selection2 == selection
    assert model2.threshold != model.threshold

    # The stored scaler is exactly the train statistics (ddof=0).
    (train_x, _), _, _ = _splits()
    np.testing.assert_array_equal(model.mean, train_x.mean(axis=0))
    np.testing.assert_array_equal(model.std, train_x.std(axis=0, ddof=0))


# 5. D29 threshold on the score scale ------------------------------------------------------------

def test_threshold_is_val_b_eer_point_of_model_score_and_predict_is_binary(config):
    model, selection, threshold_report = _fit(config)
    _, _, (val_b_x, val_b_y) = _splits()
    scores = model.score(val_b_x)
    # Bit-for-bit: the threshold is the val-B EER point of the stored model's own score.
    assert model.threshold == eer(val_b_y, scores)[1]
    assert model.threshold == fit_baseline_threshold(scores, val_b_y)
    assert threshold_report.threshold == model.threshold
    assert threshold_report.val_b_eer == eer(val_b_y, scores)[0]
    assert (threshold_report.n_spoof_val_b, threshold_report.n_bonafide_val_b) == (N_VAL_B, N_VAL_B)
    assert (model.C, model.gamma) == (selection.selected_C, selection.selected_gamma)
    # Forced binary: never "uncertain", and synthetic exactly when score >= threshold.
    labels = model.predict(val_b_x)
    assert set(labels.tolist()) <= {AUTHENTIC, SYNTHETIC}
    assert np.array_equal(labels == SYNTHETIC, scores >= model.threshold)


# 6. Determinism ---------------------------------------------------------------------------------

def test_fitting_twice_is_identical(config, tmp_path):
    model1, sel1, thr1 = _fit(config)
    model2, sel2, thr2 = _fit(config)
    assert sel1 == sel2 and sel1.to_json() == sel2.to_json()
    assert thr1 == thr2 and thr1.to_json() == thr2.to_json()
    _, _, (val_b_x, _) = _splits()
    np.testing.assert_array_equal(model1.score(val_b_x), model2.score(val_b_x))
    hash1 = artifact_sha256(save(model1, tmp_path / "a"))
    assert hash1 == artifact_sha256(save(model2, tmp_path / "b"))


def test_report_json_has_sorted_keys_and_all_rows(config):
    _, selection, _ = _fit(config)
    data = json.loads(selection.to_json())
    assert list(data) == sorted(data)
    assert len(data["rows"]) == 16
    assert list(data["rows"][0]) == ["C", "gamma", "n_support", "val_a_eer"]


def test_report_records_sklearn_version_and_svc_settings(config):
    # Provenance: the report names the scikit-learn version that fitted the grid (the pin in
    # requirements.txt) and the SVC settings every fit used, which are scikit-learn's defaults.
    pins = [line.split("==")[1].strip() for line in
            (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.startswith("scikit-learn==")]
    _, selection, _ = _fit(config)
    assert selection.sklearn_version == pins[0] == sklearn.__version__
    expected = {"kernel": "rbf", "tol": 0.001, "shrinking": True, "cache_size": 200,
                "max_iter": -1, "class_weight": None}
    assert selection.svc_settings == expected
    data = json.loads(selection.to_json())
    assert data["sklearn_version"] == pins[0]
    assert data["svc_settings"] == expected


# 7. Invalid inputs ------------------------------------------------------------------------------

def _bad(kind: str):
    (tx, ty), (ax, ay), (bx, by) = _splits()
    tx, ty = tx.copy(), ty.copy()
    if kind == "wrong_shape":
        tx = tx[:, :39]
    elif kind == "one_class":
        ty = np.zeros_like(ty)
    elif kind == "nan":
        tx[0, 0] = np.nan
    elif kind == "length_mismatch":
        ty = ty[:-1]
    elif kind == "bad_label":
        ty[0] = 2
    elif kind == "val_b_one_class":
        by = np.ones_like(by)
    return tx, ty, ax, ay, bx, by


@pytest.mark.parametrize("kind", ["wrong_shape", "one_class", "nan", "length_mismatch",
                                  "bad_label", "val_b_one_class"])
def test_invalid_inputs_raise(config, kind):
    with pytest.raises(ValueError):
        fit_baseline(*_bad(kind), config=config, version="baseline-test")


@pytest.mark.parametrize("edit", [
    ("extra_key: 1\n", "append"),
    ("decision: D37\n", "remove"),
    ("score: decision_function", "score: predict_proba"),
    ("C: [0.1, 1, 10, 100]", "C: [1, 0.1, 10, 100]"),
    ("C: [0.1, 1, 10, 100]", "C: [0.1, 1, 10, -100]"),
    ("C: [0.1, 1, 10, 100]", "C: [0.1, 1, 10, .nan]"),
    ("gamma: [0.001, 0.01, 0.025, 0.1]", "gamma: [scale]"),
    ("tie_break: [C, gamma]", "tie_break: [gamma, C]"),
])
def test_bad_config_raises(tmp_path, edit):
    text = CONFIG_PATH.read_text(encoding="utf-8")
    old, new = edit
    if new == "append":
        text += old
    elif new == "remove":
        assert old in text
        text = text.replace(old, "")
    else:
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / "baseline.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_baseline_config(path)


def test_select_refuses_non_config(config):
    train, val_a, _ = _splits()
    with pytest.raises(ValueError):
        select_baseline(*train, *val_a, config=dataclasses.asdict(config))
