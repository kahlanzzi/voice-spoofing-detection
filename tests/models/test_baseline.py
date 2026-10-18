"""MFCC/SVM baseline scorer and artifact (D7, D29, D37): synthetic 40-feature data only, seed 2026."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.svm import SVC

import src.models.baseline as baseline_module
from src.models.baseline import (
    JSON_NAME,
    N_FEATURES,
    NPZ_NAME,
    BaselineModel,
    artifact_sha256,
    fit_scaler,
    from_sklearn,
    load,
    save,
)
from src.risk.decisions import AUTHENTIC, SYNTHETIC, UNCERTAIN

BASELINE_FILE = Path(__file__).resolve().parents[2] / "src" / "models" / "baseline.py"


def _data(n_per_class: int = 100, shift: float = 0.6, offset: float = 0.0, seed: int = 2026):
    """Two Gaussian classes in 40 dimensions; spoof (1) is shifted by +shift in every feature.

    offset moves both classes, so a second split can have different statistics from train.
    Features have different scales (1..40), as raw MFCC statistics do, so scaling matters.
    """
    rng = np.random.default_rng(seed)
    scales = np.arange(1, N_FEATURES + 1, dtype=float)
    bona = rng.normal(0.0, 1.0, (n_per_class, N_FEATURES))
    spoof = rng.normal(shift, 1.0, (n_per_class, N_FEATURES))
    x = (np.vstack([bona, spoof]) + offset) * scales
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return x, y


def _fitted(threshold: float = 0.0, C: float = 1.0, gamma: float = 0.025):
    x, y = _data()
    mean, std = fit_scaler(x)
    svc = SVC(kernel="rbf", C=C, gamma=gamma).fit((x - mean) / std, y)
    model = from_sklearn(svc, mean=mean, std=std, threshold=threshold, version="baseline-test")
    return model, svc, x, y


# 1. Train-only scaler ---------------------------------------------------------------------------

def test_fit_scaler_uses_train_only():
    train, _ = _data()
    val, _ = _data(offset=2.0, seed=7)
    train_copy = train.copy()
    mean, std = fit_scaler(train)
    # The scaler is exactly the train statistics (ddof=0), and fitting does not touch the input.
    np.testing.assert_array_equal(mean, train.mean(axis=0))
    np.testing.assert_array_equal(std, train.std(axis=0, ddof=0))
    np.testing.assert_array_equal(train, train_copy)
    # Val has different statistics; the stored scaler is not refitted on it, so standardized val
    # is not centred (its mean stays near the offset of 2 standard units).
    assert not np.allclose(val.mean(axis=0), mean)
    model, _, _, _ = _fitted()
    np.testing.assert_array_equal(model.mean, mean)
    model.score(val)
    np.testing.assert_array_equal(model.mean, mean)
    np.testing.assert_array_equal(model.std, std)
    assert ((val - model.mean) / model.std).mean() > 1.0


def test_fit_scaler_rejects_bad_input():
    x, _ = _data()
    constant = x.copy()
    constant[:, 3] = 5.0  # std == 0: cannot be standardized
    with pytest.raises(ValueError, match="constant"):
        fit_scaler(constant)
    nan = x.copy()
    nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fit_scaler(nan)
    inf = x.copy()
    inf[1, 1] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        fit_scaler(inf)
    for bad in (x[:, :39], x[0], x[None], np.empty((0, N_FEATURES))):
        with pytest.raises(ValueError, match="shape"):
            fit_scaler(bad)


# 2. NumPy scorer equals scikit-learn -----------------------------------------------------------

def test_score_equals_sklearn_decision_function():
    model, svc, x, _ = _fitted()
    val, _ = _data(offset=0.5, seed=11)
    for features in (x, val):
        expected = svc.decision_function((features - model.mean) / model.std)
        # Same formula, so only rounding differences remain.
        assert np.max(np.abs(model.score(features) - expected)) <= 1e-9


@pytest.mark.parametrize("gamma", ["scale", "auto"])
def test_from_sklearn_refuses_non_numeric_gamma(gamma):
    # D37 fixes gamma as explicit numbers; a data-dependent string would hide the value used.
    x, y = _data()
    mean, std = fit_scaler(x)
    svc = SVC(kernel="rbf", C=1, gamma=gamma).fit((x - mean) / std, y)
    with pytest.raises(ValueError, match="gamma"):
        from_sklearn(svc, mean=mean, std=std, threshold=0.0, version="v")


# 3. Sign and forced-binary label ---------------------------------------------------------------

def test_spoof_scores_higher_and_predict_is_forced_binary():
    model, _, x, y = _fitted(threshold=0.0)
    s = model.score(x)
    # Higher score = more likely spoof (Protocol §5).
    assert s[y == 1].mean() > s[y == 0].mean()
    labels = model.predict(x)
    assert set(labels.tolist()) <= {SYNTHETIC, AUTHENTIC}
    assert UNCERTAIN not in labels
    # The shared band rule with half_width 0: synthetic exactly when score >= threshold.
    np.testing.assert_array_equal(labels == SYNTHETIC, s >= model.threshold)


# 4. Save / load round trip ----------------------------------------------------------------------

def test_save_load_round_trip(tmp_path):
    model, _, x, _ = _fitted(threshold=0.123)
    out = save(model, tmp_path / "baseline")
    assert sorted(p.name for p in out.iterdir()) == sorted([NPZ_NAME, JSON_NAME])
    assert not list(out.rglob("*.pkl"))
    loaded = load(out)
    np.testing.assert_array_equal(loaded.score(x), model.score(x))
    assert (loaded.version, loaded.gamma, loaded.C, loaded.intercept, loaded.threshold) == (
        model.version, model.gamma, model.C, model.intercept, model.threshold,
    )
    # The arrays load without pickle.
    with np.load(out / NPZ_NAME, allow_pickle=False) as npz:
        assert set(npz.files) == {"mean", "std", "support_vectors", "dual_coef"}


def test_save_is_byte_deterministic(tmp_path):
    # Fixed zip metadata: the same model saved twice gives identical bytes and the same F4 hash.
    model, _, x, _ = _fitted(threshold=0.5)
    a = save(model, tmp_path / "a")
    b = save(model, tmp_path / "b")
    for name in (NPZ_NAME, JSON_NAME):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    assert artifact_sha256(a) == artifact_sha256(b)
    np.testing.assert_array_equal(load(b).score(x), model.score(x))


def test_save_refuses_non_empty_folder(tmp_path):
    model, _, _, _ = _fitted()
    (tmp_path / "other.txt").write_text("x")
    with pytest.raises(ValueError, match="new or empty"):
        save(model, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["other.txt"]


# 5. Tampering -----------------------------------------------------------------------------------

def test_tampering_changes_artifact_hash(tmp_path):
    model, _, _, _ = _fitted()
    out = save(model, tmp_path / "a")
    original = artifact_sha256(out)
    assert artifact_sha256(out) == original  # deterministic for the same files

    # One value changed in the arrays.
    with np.load(out / NPZ_NAME, allow_pickle=False) as npz:
        arrays = {k: npz[k].copy() for k in npz.files}
    arrays["dual_coef"][0] += 1e-6
    np.savez(out / NPZ_NAME, **arrays)
    after_npz = artifact_sha256(out)
    assert after_npz != original

    # One value changed in the JSON.
    meta = json.loads((out / JSON_NAME).read_text(encoding="utf-8"))
    meta["threshold"] += 1e-6
    (out / JSON_NAME).write_text(json.dumps(meta), encoding="utf-8")
    assert artifact_sha256(out) not in {original, after_npz}


@pytest.mark.parametrize("change", ["rename_json_key", "extra_json_key", "bad_schema", "bad_n_features",
                                    "missing_npz_key", "wrong_shape", "nan_value", "object_array",
                                    "corrupt_npz", "missing_file"])
def test_load_rejects_corrupted_artifact(tmp_path, change):
    model, _, _, _ = _fitted()
    out = save(model, tmp_path / "a")
    meta = json.loads((out / JSON_NAME).read_text(encoding="utf-8"))
    with np.load(out / NPZ_NAME, allow_pickle=False) as npz:
        arrays = {k: npz[k].copy() for k in npz.files}

    if change == "rename_json_key":
        meta["thresh"] = meta.pop("threshold")
    elif change == "extra_json_key":
        meta["probability"] = True
    elif change == "bad_schema":
        meta["schema_version"] = 2
    elif change == "bad_n_features":
        meta["n_features"] = 39
    elif change == "missing_npz_key":
        del arrays["std"]
    elif change == "wrong_shape":
        arrays["mean"] = arrays["mean"][:39]
    elif change == "nan_value":
        arrays["support_vectors"][0, 0] = np.nan
    elif change == "object_array":
        arrays["dual_coef"] = np.array([{"x": 1}], dtype=object)  # pickled inside the npz
    (out / JSON_NAME).write_text(json.dumps(meta), encoding="utf-8")
    np.savez(out / NPZ_NAME, **arrays)
    if change == "corrupt_npz":
        (out / NPZ_NAME).write_bytes(b"not an npz file")
    elif change == "missing_file":
        (out / JSON_NAME).unlink()

    with pytest.raises(ValueError):
        load(out)


# 6. from_sklearn refuses other models ----------------------------------------------------------

def test_from_sklearn_refuses_other_classes_and_kernels():
    x, y = _data()
    mean, std = fit_scaler(x)
    z = (x - mean) / std
    kw = dict(mean=mean, std=std, threshold=0.0, version="v")
    # Labels 1/2: the sign convention (positive = spoof = 1) would be unknown.
    with pytest.raises(ValueError, match="classes_"):
        from_sklearn(SVC(kernel="rbf").fit(z, y + 1), **kw)
    # Three classes: not a binary SVC.
    y3 = y.copy()
    y3[:10] = 2
    with pytest.raises(ValueError, match="classes_"):
        from_sklearn(SVC(kernel="rbf").fit(z, y3), **kw)
    for kernel in ("linear", "poly", "sigmoid"):
        with pytest.raises(ValueError, match="rbf"):
            from_sklearn(SVC(kernel=kernel).fit(z, y), **kw)
    with pytest.raises(ValueError, match="not fitted"):
        from_sklearn(SVC(kernel="rbf"), **kw)


def test_model_rejects_invalid_fields():
    model, _, _, _ = _fitted()
    good = dict(version="v", mean=model.mean, std=model.std, support_vectors=model.support_vectors,
                dual_coef=model.dual_coef, intercept=model.intercept, gamma=model.gamma, C=model.C,
                threshold=0.0)
    for field, value in [("version", ""), ("gamma", 0.0), ("C", -1.0), ("threshold", np.inf),
                         ("intercept", True), ("std", np.zeros(N_FEATURES)),
                         ("dual_coef", model.dual_coef[:-1])]:
        with pytest.raises(ValueError):
            BaselineModel(**{**good, field: value})
    with pytest.raises(ValueError, match="shape"):
        model.score(np.zeros((3, 39)))


# 7. Layering ------------------------------------------------------------------------------------

def _imported_modules(path: Path) -> list[str]:
    """Absolute module names imported by a source file (relative imports resolved inside src.models)."""
    names = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # from .x / ..x: resolve against the src.models package
                base = ["src", "models"][: max(0, 2 - (node.level - 1))]
                names.append(".".join(base + ([node.module] if node.module else [])))
            else:
                names.append(node.module or "")
    return names


def test_baseline_does_not_import_research_code_or_sklearn():
    # Stage 13 scores with this module; it has no Evaluation or Tuning service, and scoring must not
    # need scikit-learn (the scorer is NumPy only).
    names = _imported_modules(BASELINE_FILE)
    assert names, "no imports parsed"
    offending = [
        n for n in names
        for research in ("evaluation", "tuning")
        if n == f"src.{research}" or n.startswith(f"src.{research}.") or n.startswith(research)
    ]
    assert offending == []
    assert not any(n == "sklearn" or n.startswith("sklearn.") for n in names)
    assert not any(n in {"pickle", "joblib", "dill", "cloudpickle"} for n in names)
    assert baseline_module.N_FEATURES == 40
