"""MFCC/SVM baseline: train-only scaler, NumPy RBF-SVM scorer and a non-pickle artifact (D7, D29, D37).

The baseline is evaluation-only (D7): it is scored on val-A, val-B, test and, at Stage 13, on the
external windows, but it is never deployed and has no risk policy. Fitting (the D37 grid search)
lives in src/tuning/baseline_fit.py (step B2); this module only stores and applies a fitted model.

Conventions (Protocol v1.0 §3, §5, §11; D4, D37):
- features: an n x 40 float array, the MFCC mean + std of 20 coefficients from the frozen profile.
- scaler: per-feature mean and std (ddof=0) fitted on the FoR train split only, stored in the
  artifact and applied unchanged to every other split (D37).
- score: the RBF-SVM decision function f(x) = sum_i dual_coef_i * exp(-gamma * ||sv_i - x||^2) +
  intercept on the standardized features; higher score = more likely spoof (labels 1 = spoof).
- label: forced binary at the D29 threshold through src.risk.decisions.three_way_decisions with
  half_width = 0, the single band rule; the baseline never says "uncertain".

Why no pickle or joblib: a pickled scikit-learn model can execute code when it is loaded, the same
risk docs/model_bundle.md names for PyTorch checkpoints. The artifact is a folder with baseline.npz
(numeric arrays only, loaded with allow_pickle=False) and baseline.json (scalars, strict keys), so
loading it never runs code. The scorer is re-implemented in NumPy, so Stage 13 scoring does not
need scikit-learn at all; tests check it against scikit-learn's decision_function.

Layering: Stage 13 imports this module to score, so it must not import src.evaluation or
src.tuning (research code). It may use the shared core (src.risk, src.bundle).
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np

from src.bundle.bundle import sha256_file
from src.risk.decisions import three_way_decisions

N_FEATURES = 40  # MFCC mean + std of 20 coefficients (Protocol §3, §11)
SCHEMA_VERSION = 1
NPZ_NAME = "baseline.npz"
JSON_NAME = "baseline.json"

_NPZ_KEYS = {"mean", "std", "support_vectors", "dual_coef"}
_JSON_KEYS = {"schema_version", "version", "gamma", "C", "intercept", "threshold", "n_features"}


def _features(x, name: str) -> np.ndarray:
    """Return x as an n x 40 float array of finite values (n >= 1), or raise ValueError."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[1] != N_FEATURES or a.shape[0] == 0:
        raise ValueError(f"{name} must have shape (n, {N_FEATURES}) with n >= 1, got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains {int((~np.isfinite(a)).sum())} NaN or infinite values")
    return a


def _finite_number(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _positive_number(name: str, value) -> float:
    v = _finite_number(name, value)
    if v <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return v


def _frozen_array(name: str, value, shape: tuple) -> np.ndarray:
    """A read-only float64 copy of value with the given shape (None = any length), finite values."""
    a = np.array(value, dtype=float)
    if a.ndim != len(shape) or any(s is not None and s != d for s, d in zip(shape, a.shape)):
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    a.setflags(write=False)
    return a


def fit_scaler(train_features) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and std (ddof=0) of the FoR train features only (D37).

    A constant feature (std == 0) cannot be standardized and raises ValueError.
    """
    x = _features(train_features, "train_features")
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    constant = np.flatnonzero(std == 0)
    if constant.size:
        raise ValueError(f"constant train features (std == 0) at indices {constant.tolist()}")
    return mean, std


@dataclass(frozen=True, eq=False)
class BaselineModel:
    """A fitted RBF-SVM with its train-only scaler and its D29 threshold (score scale)."""

    version: str
    mean: np.ndarray
    std: np.ndarray
    support_vectors: np.ndarray
    dual_coef: np.ndarray
    intercept: float
    gamma: float
    C: float  # not used to score; recorded for provenance (D37 grid)
    threshold: float

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError(f"version must be a non-empty string, got {self.version!r}")
        set_ = object.__setattr__  # frozen dataclass: fields are set once, here
        set_(self, "mean", _frozen_array("mean", self.mean, (N_FEATURES,)))
        set_(self, "std", _frozen_array("std", self.std, (N_FEATURES,)))
        if (self.std <= 0).any():
            raise ValueError("std must be > 0 for every feature")
        set_(self, "support_vectors",
             _frozen_array("support_vectors", self.support_vectors, (None, N_FEATURES)))
        if self.support_vectors.shape[0] == 0:
            raise ValueError("support_vectors is empty")
        set_(self, "dual_coef",
             _frozen_array("dual_coef", self.dual_coef, (self.support_vectors.shape[0],)))
        set_(self, "intercept", _finite_number("intercept", self.intercept))
        set_(self, "gamma", _positive_number("gamma", self.gamma))
        set_(self, "C", _positive_number("C", self.C))
        set_(self, "threshold", _finite_number("threshold", self.threshold))

    def score(self, features) -> np.ndarray:
        """Continuous SVM score per row (higher = more likely spoof), same as SVC.decision_function."""
        z = (_features(features, "features") - self.mean) / self.std
        sv = self.support_vectors
        # ||sv - z||^2 = ||z||^2 + ||sv||^2 - 2 z.sv; clipped at 0 against rounding.
        sq = (z * z).sum(axis=1)[:, None] + (sv * sv).sum(axis=1)[None, :] - 2.0 * (z @ sv.T)
        kernel = np.exp(-self.gamma * np.maximum(sq, 0.0))
        return kernel @ self.dual_coef + self.intercept

    def predict(self, features) -> np.ndarray:
        """Forced-binary labels ("synthetic" / "authentic") at the D29 threshold; never "uncertain"."""
        return three_way_decisions(self.score(features), self.threshold, 0.0)


def from_sklearn(svc, *, mean, std, threshold, version: str) -> BaselineModel:
    """Copy a fitted sklearn.svm.SVC(kernel="rbf") with classes [0, 1] into a BaselineModel.

    With classes [0, 1], SVC.decision_function is positive towards class 1 (spoof), so the copied
    dual_coef_ and intercept_ give the same sign convention. gamma is the numeric value the SVC
    used (_gamma, which resolves "scale" / "auto"), not the constructor argument.
    """
    if getattr(svc, "kernel", None) != "rbf":
        raise ValueError(f"svc must use kernel='rbf', got {getattr(svc, 'kernel', None)!r}")
    classes = getattr(svc, "classes_", None)
    if classes is None:
        raise ValueError("svc is not fitted (no classes_)")
    if np.asarray(classes).tolist() != [0, 1]:
        raise ValueError(f"svc.classes_ must be [0, 1] (0 = bonafide, 1 = spoof), got {classes!r}")
    dual = np.asarray(svc.dual_coef_)
    if dual.shape[0] != 1:
        raise ValueError(f"svc.dual_coef_ must have one row (binary SVC), got shape {dual.shape}")
    return BaselineModel(
        version=version,
        mean=mean,
        std=std,
        support_vectors=svc.support_vectors_,
        dual_coef=dual[0],
        intercept=float(np.asarray(svc.intercept_)[0]),
        gamma=float(svc._gamma),
        C=float(svc.C),
        threshold=threshold,
    )


def save(model: BaselineModel, directory) -> Path:
    """Write baseline.npz and baseline.json into a new or empty folder; return the folder.

    A non-empty folder is refused, so a frozen artifact is never overwritten. If writing fails, the
    files written so far are removed.
    """
    if not isinstance(model, BaselineModel):
        raise ValueError(f"model must be a BaselineModel, got {type(model).__name__}")
    directory = Path(directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError(f"{directory} must be a new or empty folder")
    meta = {
        "schema_version": SCHEMA_VERSION,
        "version": model.version,
        "gamma": model.gamma,
        "C": model.C,
        "intercept": model.intercept,
        "threshold": model.threshold,
        "n_features": N_FEATURES,
    }
    text = json.dumps(meta, indent=2, sort_keys=True, allow_nan=False) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    try:
        npz_path = directory / NPZ_NAME
        written.append(npz_path)
        np.savez(
            npz_path,
            mean=model.mean,
            std=model.std,
            support_vectors=model.support_vectors,
            dual_coef=model.dual_coef,
        )
        json_path = directory / JSON_NAME
        written.append(json_path)
        json_path.write_text(text, encoding="utf-8")
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return directory


def _check_keys(where: str, keys, expected: set[str]) -> None:
    missing = sorted(expected - set(keys))
    unknown = sorted(set(keys) - expected)
    if missing or unknown:
        raise ValueError(f"{where}: missing keys {missing}, unknown keys {unknown}")


def load(directory) -> BaselineModel:
    """Read and verify a saved baseline; raise ValueError on any missing, extra or invalid entry.

    The arrays are read with allow_pickle=False, so an object array (pickle) is refused, not run.
    Shapes and finiteness are checked by BaselineModel itself.
    """
    directory = Path(directory)
    npz_path, json_path = directory / NPZ_NAME, directory / JSON_NAME
    for path in (npz_path, json_path):
        if not path.is_file():
            raise ValueError(f"missing baseline file {path}")

    try:
        meta = json.loads(json_path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{json_path} is not valid JSON: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"{json_path} must hold a JSON object")
    _check_keys(JSON_NAME, meta.keys(), _JSON_KEYS)
    if meta["schema_version"] != SCHEMA_VERSION or isinstance(meta["schema_version"], bool):
        raise ValueError(f"unsupported schema_version {meta['schema_version']!r}")
    if meta["n_features"] != N_FEATURES or isinstance(meta["n_features"], bool):
        raise ValueError(f"n_features must be {N_FEATURES}, got {meta['n_features']!r}")

    try:
        with np.load(io.BytesIO(npz_path.read_bytes()), allow_pickle=False) as npz:
            _check_keys(NPZ_NAME, npz.files, _NPZ_KEYS)
            arrays = {key: npz[key] for key in _NPZ_KEYS}
    except (zipfile.BadZipFile, OSError, EOFError) as exc:
        raise ValueError(f"{npz_path} is not a readable .npz file: {exc}") from exc
    for key, value in arrays.items():
        if value.dtype.kind != "f":
            raise ValueError(f"{NPZ_NAME}[{key}] must be a float array, got dtype {value.dtype}")

    return BaselineModel(
        version=meta["version"],
        mean=arrays["mean"],
        std=arrays["std"],
        support_vectors=arrays["support_vectors"],
        dual_coef=arrays["dual_coef"],
        intercept=meta["intercept"],
        gamma=meta["gamma"],
        C=meta["C"],
        threshold=meta["threshold"],
    )


def artifact_sha256(directory) -> str:
    """One SHA-256 for the artifact, for the F4 record (D36).

    It hashes the ASCII text "baseline.json:<sha256>\\nbaseline.npz:<sha256>\\n" (files in this
    fixed order), so a change to either file changes it.
    """
    directory = Path(directory)
    lines = []
    for name in (JSON_NAME, NPZ_NAME):
        path = directory / name
        if not path.is_file():
            raise ValueError(f"missing baseline file {path}")
        lines.append(f"{name}:{sha256_file(path)}\n")
    return hashlib.sha256("".join(lines).encode("ascii")).hexdigest()
