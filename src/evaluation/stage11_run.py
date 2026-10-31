"""Official Stage 11 FoR test run and its validation-only dry run (Protocol §7; D29, D31, D41, D45, D46).

The official run scores the FoR test split once with the four models of the D41 model set:
baseline-v1 and the three CNN seeds of cnn-v1, all on CPU (D46 (1)). It has two phases, two CLI
calls, so the notebook can flush the Drive mount between them: a re-read on the same mount does not
prove that the claim reached Drive (D46 (6)). In the D46 (5) order:

claim() (no test audio is read):
1. Pre-claim checks, nothing written: results_ref and output folder, clean full-SHA commit,
   environment gate (D46 (2): Python and torch versions), the committed Stage 7, 8 and 10 evidence,
   F2 manifest and F3 profile hashes, every model loaded against its recorded hash, the
   reproduction gate (D46 (4): two val-A cache builds per feature, byte-identical and equal to
   Stage 7 / Stage 8), every test row valid under the clip_scores row rules, and no earlier
   ("for", "test", "stage11_test") record in the registry.
2. RunRegistry.claim (D31, D41), re-read from the file and verified.
3. Environment evidence: out/ is created with environment.json (the D46 (3) record) and
   pip_freeze.txt, then out/claim.json (run_id, commit, registry SHA-256 after the claim,
   results_ref). The run_id is returned.

score(run_id) (after the notebook has flushed and re-mounted Drive):
4. out/claim.json must name this run_id and the current commit (local files and git only, no test
   access); a wrong run_id is refused without any write. Then out/scoring_started.json is created,
   exclusively: a second score call is refused, even after a crash before any audio read (D46 (8):
   no rerun after the claim). Right after it, out/environment_score.json records the environment
   of the scoring session (recorded, not asserted).
5. The pre-claim checks again; the registry's single ("for", "test", "stage11_test") record must
   be this run_id, started, same commit, and the registry and out/ must still match claim.json.
6. Test features in memory, never persisted (D46 (7)): one read_verified_audio + extract_features
   call per clip gives the log-Mel and the MFCC; computed twice, the two canonical feature hashes
   must match.
7. Every model scored twice on CPU, byte-identical; bundle-seed p and prediction from the frozen
   risk policy, baseline prediction from its D29 threshold; every prediction recomputed
   independently before anything is written (Entry 99).
8. Per-clip score files (D41, D45) via write_clip_scores, then run_report.json; no metrics (they
   are computed from the stored files in a later step).
9. RunRegistry.complete with the results_ref of claim.json.
Any failure after the claim raises and leaves the started record: no retry, no cleanup, and the
registry is never rewritten (D46 (8)).

rehearse() is the D46 dry run on val-A and val-B only: the same pre-claim checks (no registry), the
same feature and scoring functions, and known answers from the caches, the checkpoints, Stage 7 and
Stage 10. It refuses every other split before anything is read. If any check fails, it still
writes rehearsal_report.json with every check done so far and "passed": false, then raises.

Expected hashes come from the committed evidence files, never from constants, except the D41
baseline artifact hash, which must also equal Stage 7's. There is no split argument: the test
split is fixed in code (Entry 99). Nothing is logged to W&B. This module never caches test
features, never fits, tunes or recalibrates anything, and never touches DEEP-VOICE.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.evaluation.clip_scores import (
    LABELS,
    _check_choice,
    _check_clip_id,
    _check_text,
    _create,
    write_clip_scores,
)
from src.evaluation.metrics import eer
from src.evaluation.run_registry import STARTED, RunRegistry
from src.features.logmel_cache import FEATURE_SHAPE as LOG_MEL_SHAPE
from src.features.logmel_cache import load_logmel_cache
from src.features.mfcc_cache import LABEL_CODES, _split_rows, load_mfcc_cache
from src.freezing.manifests import safe_relative_path
from src.models.baseline import N_FEATURES, artifact_sha256
from src.models.baseline import load as load_baseline
from src.models.cnn_checkpoint import checkpoint_sha256, load_checkpoint
from src.preprocessing.features import extract_features
from src.preprocessing.profile import load_profile
from src.preprocessing.validation import (
    F2_MANIFEST_SHA256,
    code_version,
    load_f2_manifest,
    read_verified_audio,
    sha256_file,
)
from src.risk.decisions import AUTHENTIC, SYNTHETIC, UNCERTAIN
from src.risk.policy import RiskPolicy
from src.training.cnn_config import load_cnn_config
from src.tuning.baseline_run import _json_text
from src.tuning.risk_run import bundle_checkpoint, score_clips

SCHEMA_VERSION = 1
DATASET = "for"
SPLIT = "test"
PURPOSE = "stage11_test"
SEEDS = (0, 1, 2)
BASELINE_ID = "baseline-v1"
CNN_IDS = {seed: f"cnn-v1-seed{seed}" for seed in SEEDS}
MODEL_IDS = (BASELINE_ID, *CNN_IDS.values())
# D41: the baseline-v1 artifact hash; Stage 7's committed artifact_sha256 must equal it too.
BASELINE_ARTIFACT_SHA256 = "42cc4d53a7cd881f56e4fa54fcf4797de8885b404d0a2b31c293555c868df10a"
DEVICE = "cpu"  # D46 (1)
# D46 (2) environment gate (asserted).
REQUIRED_PYTHON = "3.13"
REQUIRED_TORCH = "2.11.0"
ROLE_BASELINE = "baseline"
ROLE_BUNDLE = "cnn_bundle_seed"
ROLE_OTHER = "cnn_other_seed"
REHEARSAL_SPLITS = ("val-A", "val-B")
GATE_SPLIT = "val-A"  # D46 (4): the reproduction gate rebuilds val-A only
LOG_MEL, MFCC = "log_mel", "mfcc"  # cache sub-folders and feature names
CACHE_LOADERS = {LOG_MEL: load_logmel_cache, MFCC: load_mfcc_cache}
FEATURE_SHAPES = {LOG_MEL: LOG_MEL_SHAPE, MFCC: (N_FEATURES,)}
CLIP_SCORES_DIR = "clip_scores"
RUN_REPORT = "run_report.json"
REHEARSAL_REPORT = "rehearsal_report.json"
PIP_FREEZE = "pip_freeze.txt"
ENVIRONMENT_JSON = "environment.json"
CLAIM_JSON = "claim.json"
SCORING_MARKER = "scoring_started.json"
ENVIRONMENT_SCORE_JSON = "environment_score.json"
STAGE_REPORT = "run_report.json"  # the Stage 7, 8 and 10 evidence file name
ENV_PACKAGES = ("numpy", "scipy", "librosa", "soxr", "numba", "llvmlite", "soundfile", "scikit-learn",
                "torch")
CPUINFO = Path("/proc/cpuinfo")
UNAVAILABLE = "unavailable"
LABEL_NAMES = {code: name for name, code in LABEL_CODES.items()}
_COMMIT = re.compile(r"[0-9a-f]{40}")
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_ROW_COLUMNS = ("path", "label", "subgroup", "group", "split", "hash")
_VAL_B_SCORE_KEYS = {"labels", "p", "paths", "score"}


class CheckFailed(ValueError):
    """A recorded check failed; key names it (the rehearsal report records it)."""

    def __init__(self, key: str, message: str):
        super().__init__(message)
        self.key = key


def _record(checks: dict, key: str, expected, actual, passed: bool) -> None:
    """Record expected/actual/passed under key; raise CheckFailed if not passed."""
    checks[key] = {"expected": expected, "actual": actual, "passed": passed}
    if not passed:
        raise CheckFailed(key, f"check failed: {key}: expected {expected!r}, got {actual!r}")


def _check(checks: dict, key: str, expected, actual) -> None:
    """_record with passed = (actual == expected); a bool never equals a number."""
    same = isinstance(actual, bool) == isinstance(expected, bool) and actual == expected
    _record(checks, key, expected, actual, bool(same))


def _read_json(path: Path) -> dict:
    data = json.loads(Path(path).read_bytes().decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return data


def _same(name: str, values: dict) -> str:
    """The one value shared by every source, or raise naming them."""
    if len(set(values.values())) != 1:
        raise ValueError(f"evidence disagrees on {name}: {values}")
    return next(iter(values.values()))


# --- Evidence -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Evidence:
    """The expected values, read from the committed Stage 7, 8 and 10 evidence files."""

    bundle_seed: int
    checkpoint_sha256: dict  # seed (int) -> checkpoint_sha256
    baseline_artifact_sha256: str
    baseline_threshold: float
    baseline_val_a_eer: float
    baseline_val_b_eer: float
    cache_npz_sha256: dict  # (feature, split) -> npz_sha256
    cnn_config_sha256: str
    risk_policy_sha256: str
    val_b_scores_sha256: str
    batch_size: int
    profile_fingerprint: str
    profile_config_sha256: str
    f2_manifest_sha256: str


def load_evidence(stage7_dir, stage8_dir, stage10_dir) -> Evidence:
    """Read and cross-check the committed evidence; raise ValueError on any disagreement."""
    s7 = _read_json(Path(stage7_dir) / STAGE_REPORT)
    s8 = _read_json(Path(stage8_dir) / STAGE_REPORT)
    s10 = _read_json(Path(stage10_dir) / STAGE_REPORT)

    bundle_seed, bundle_sha = bundle_checkpoint(stage8_dir)  # run_report and selection_report agree
    hashes = s8["checkpoint_sha256"]
    if not isinstance(hashes, dict) or sorted(hashes) != sorted(str(s) for s in SEEDS):
        raise ValueError(f"Stage 8 checkpoint_sha256 must name seeds {list(SEEDS)}, got {hashes!r}")
    if bundle_seed not in SEEDS:
        raise ValueError(f"Stage 8 bundle seed {bundle_seed} is not one of {SEEDS}")
    if s10["bundle_seed"] != bundle_seed or s10["checkpoint_sha256"] != bundle_sha:
        raise ValueError(
            f"Stage 10 policy was fitted for seed {s10['bundle_seed']!r} / checkpoint "
            f"{s10['checkpoint_sha256']!r}; Stage 8 bundle is seed {bundle_seed} / {bundle_sha}"
        )
    if s8["caches"]["val-A"]["npz_sha256"] != s8["val_a_cache_npz_sha256"]:
        raise ValueError("Stage 8 run_report disagrees with itself on the val-A cache npz_sha256")
    if s7["version"] != BASELINE_ID:
        raise ValueError(f"Stage 7 version is {s7['version']!r}, expected {BASELINE_ID!r}")
    if s7["artifact_sha256"] != BASELINE_ARTIFACT_SHA256:
        raise ValueError(f"Stage 7 artifact_sha256 {s7['artifact_sha256']} differs from the D41 value "
                         f"{BASELINE_ARTIFACT_SHA256}")
    if s10["scoring"]["device"] != DEVICE:
        raise ValueError(f"Stage 10 scored on {s10['scoring']['device']!r}, Stage 11 scores on {DEVICE!r}")
    sources = {"stage7": s7, "stage8": s8, "stage10": s10}
    return Evidence(
        bundle_seed=bundle_seed,
        checkpoint_sha256={seed: hashes[str(seed)] for seed in SEEDS},
        baseline_artifact_sha256=s7["artifact_sha256"],
        baseline_threshold=s7["val_b_threshold"],
        baseline_val_a_eer=s7["val_a_eer"],
        baseline_val_b_eer=s7["val_b_eer"],
        cache_npz_sha256={
            (LOG_MEL, "val-A"): s8["caches"]["val-A"]["npz_sha256"],
            (LOG_MEL, "val-B"): s10["caches"]["val-B"]["npz_sha256"],
            (MFCC, "val-A"): s7["caches"]["val-A"]["npz_sha256"],
            (MFCC, "val-B"): s7["caches"]["val-B"]["npz_sha256"],
        },
        cnn_config_sha256=s8["cnn_config_sha256"],
        risk_policy_sha256=s10["risk_policy_sha256"],
        val_b_scores_sha256=s10["val_b_scores_sha256"],
        batch_size=s10["scoring"]["batch_size"],
        **{key: _same(key, {name: s[key] for name, s in sources.items()})
           for key in ("profile_fingerprint", "profile_config_sha256", "f2_manifest_sha256")},
    )


# --- Environment ----------------------------------------------------------------------------------

def environment_gate(checks: dict) -> None:
    """D46 (2), asserted: Python 3.13.x and torch 2.11.0 (any local build suffix, e.g. +cu130)."""
    _check(checks, "environment.python", REQUIRED_PYTHON, ".".join(platform.python_version_tuple()[:2]))
    _check(checks, "environment.torch", REQUIRED_TORCH, torch.__version__.split("+")[0])


def _cpu_info(failures: dict) -> dict[str, Any]:
    try:
        text = CPUINFO.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        failures["cpu"] = f"{type(error).__name__}: {error}"
        return {"model": UNAVAILABLE, "flags": UNAVAILABLE, "avx512": UNAVAILABLE}
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields.setdefault(key.strip(), value.strip())  # first processor only
    flags = fields.get("flags")
    if flags is None:
        failures["cpu_flags"] = f"no 'flags' line in {CPUINFO}"
    names = sorted(flags.split()) if flags is not None else None
    return {
        "model": fields.get("model name", UNAVAILABLE),
        "flags": " ".join(names) if names is not None else UNAVAILABLE,
        "avx512": [f for f in names if f.startswith("avx512")] if names is not None else UNAVAILABLE,
    }


def _gpu_info(failures: dict) -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except FileNotFoundError:
        return "none"  # no NVIDIA driver installed
    except (OSError, subprocess.SubprocessError) as error:
        failures["gpu"] = f"{type(error).__name__}: {error}"
        return "none"
    return out or "none"


def environment_record() -> tuple[dict[str, Any], str]:
    """D46 (3), recorded, not asserted: (record, pip freeze text). Never raises: failures are recorded.

    The record holds the pip freeze SHA-256 (the text itself goes to pip_freeze.txt), the versions of
    ENV_PACKAGES, the CPU model and flags (incl. AVX-512) from /proc/cpuinfo, GPU name and driver
    from nvidia-smi ("none" without one), platform.platform() and the Python version.
    """
    failures: dict[str, str] = {}
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True,
            timeout=300,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        failures["pip_freeze"] = f"{type(error).__name__}: {error}"
        freeze = ""
    versions = {}
    for name in ENV_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except Exception as error:  # any failure is recorded, never raised (D46 (3))
            failures[f"version:{name}"] = f"{type(error).__name__}: {error}"
            versions[name] = UNAVAILABLE
    record = {
        "pip_freeze_sha256": hashlib.sha256(freeze.encode("utf-8")).hexdigest(),
        "packages": versions,
        "cpu": _cpu_info(failures),
        "gpu": _gpu_info(failures),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "failures": failures,
    }
    return record, freeze


# --- Features -------------------------------------------------------------------------------------

class _ArrayDigest:
    """Canonical SHA-256 of one feature array, fed one clip at a time (the single definition).

    Hashes the header line {"dtype": "<f4", "name": name, "shape": [n, *clip_shape]} (canonical
    JSON, sorted keys, compact separators) + "\\n", then the C-order little-endian float32 bytes of
    each clip in path order: the same digest as the whole stacked array's C-order bytes.
    """

    def __init__(self, name: str, n: int):
        self.name, self.n, self.shape, self.count = name, n, FEATURE_SHAPES[name], 0
        header = {"dtype": np.dtype("<f4").str, "name": name, "shape": [n, *self.shape]}
        self._hash = hashlib.sha256((json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n")
                                    .encode("utf-8"))

    def update(self, clip: np.ndarray) -> None:
        if clip.dtype != np.float32 or clip.shape != self.shape or not np.isfinite(clip).all():
            raise ValueError(f"{self.name} must be finite float32 {self.shape}, "
                             f"got {clip.dtype} {clip.shape}")
        self._hash.update(np.ascontiguousarray(clip, dtype="<f4").tobytes())
        self.count += 1

    def hexdigest(self) -> str:
        if self.count != self.n:
            raise ValueError(f"{self.name}: digest fed {self.count} clips, expected {self.n}")
        return self._hash.hexdigest()


def array_sha256(name: str, array: np.ndarray) -> str:
    """_ArrayDigest of a stacked (n, *clip_shape) array."""
    digest = _ArrayDigest(name, len(array))
    for clip in array:
        digest.update(clip)
    return digest.hexdigest()


def features_sha256(paths, labels, arrays: dict[str, str]) -> str:
    """SHA-256 of canonical JSON {"arrays": {name: array digest}, "labels": [...], "paths": [...]}."""
    payload = {"arrays": arrays, "labels": list(labels), "paths": list(paths)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def split_rows(manifest, split: str):
    """F2 rows of split sorted by path (mfcc_cache rules), each valid under the clip_scores row rules.

    Reuses clip_scores' own validators, so a row that write_clip_scores would refuse is refused
    here, before the claim: canonical clip_id (the path), label in {FAKE, REAL}, group a non-empty
    trimmed string (load_f2_manifest reads every column as str; a float or NaN means another
    loader was used). The subgroup column must exist (counts).
    """
    missing = [c for c in _ROW_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(f"manifest is missing columns {missing}")
    rows = _split_rows(manifest, split)
    for number, (path, label, group) in enumerate(zip(rows["path"], rows["label"], rows["group"]), start=1):
        _check_clip_id(number, path)
        _check_choice(f"row {number}: label", label, LABELS)
        _check_text("group_id", number, group)
    return rows


def _feature_pass(root, rows, profile, *, keep: bool):
    """One pass over rows: per clip one read_verified_audio + extract_features call."""
    n = len(rows)
    digests = {name: _ArrayDigest(name, n) for name in FEATURE_SHAPES}
    arrays = {name: np.empty((n, *shape), dtype=np.float32) for name, shape in FEATURE_SHAPES.items()} \
        if keep else None
    for i, (_, row) in enumerate(rows.iterrows()):
        waveform, sample_rate = read_verified_audio(root, row)
        _, log_mel, mfcc = extract_features(waveform, sample_rate, profile)
        for name, clip in ((LOG_MEL, log_mel), (MFCC, mfcc)):
            digests[name].update(clip)
            if keep:
                arrays[name][i] = clip
    hashes = {name: d.hexdigest() for name, d in digests.items()}
    paths, labels = rows["path"].tolist(), rows["label"].tolist()
    return arrays, {"arrays": hashes, "features_sha256": features_sha256(paths, labels, hashes)}


def compute_features(root, rows, profile) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Features in memory, computed twice (D46 (7)); pass 2 keeps only hashes. Raise if they differ."""
    arrays, first = _feature_pass(root, rows, profile, keep=True)
    _, second = _feature_pass(root, rows, profile, keep=False)
    if first != second:
        raise ValueError(f"feature passes differ: pass 1 {first}, pass 2 {second}")
    return arrays, {**first, "n_clips": len(rows)}


# --- Models and scoring ---------------------------------------------------------------------------

@dataclass
class Models:
    cnn: dict  # seed -> (model, mean, std, metadata)
    baseline: Any
    policy: RiskPolicy
    bundle_seed: int
    batch_size: int
    artifact_sha256: dict  # model_id -> hash


def score_models(models: Models, features: dict[str, np.ndarray], model_ids) -> dict[str, np.ndarray]:
    """Raw float64 scores per model id on CPU, each computed twice; raise unless byte-identical."""
    seeds = {CNN_IDS[s]: s for s in SEEDS}
    out = {}
    for model_id in model_ids:
        def once():
            if model_id == BASELINE_ID:
                return np.asarray(models.baseline.score(features[MFCC]), dtype=np.float64)
            model, mean, std, _ = models.cnn[seeds[model_id]]
            return score_clips(model, mean, std, features[LOG_MEL], models.batch_size)
        first, second = once(), once()
        if first.dtype != np.float64 or first.tobytes() != second.tobytes():
            raise ValueError(f"{model_id}: scores differ between two CPU scoring passes")
        out[model_id] = first
    return out


def bundle_outputs(policy: RiskPolicy, scores: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """p and prediction through RiskPolicy.decide, then checked independently (Entry 99)."""
    decisions = policy.decide(scores)
    p = np.array([d.p for d in decisions], dtype=np.float64)
    prediction = [d.prediction for d in decisions]
    if p.tobytes() != policy.calibration.transform(scores).tobytes():
        raise ValueError("bundle seed: p differs from calibration.transform of the raw scores")
    # The convention of src/risk/decisions.py three_way_decisions, written out here so the check
    # does not call the code it checks:
    #   - synthetic if p >= threshold + half_width
    #   - uncertain if threshold - half_width <= p < threshold + half_width
    #   - authentic if p < threshold - half_width
    # with the bounds computed in floating point, as there.
    upper, lower = policy.threshold + policy.half_width, policy.threshold - policy.half_width
    expected = np.where(p >= upper, SYNTHETIC, np.where(p < lower, AUTHENTIC, UNCERTAIN)).tolist()
    _require_same_predictions("bundle seed", expected, prediction)
    return p, prediction


def baseline_outputs(baseline, mfcc: np.ndarray, scores: np.ndarray) -> list[str]:
    """D29 prediction through BaselineModel.predict, checked against the raw score (Entry 99).

    Independent check, with the tie rule of three_way_decisions at half_width 0: score ==
    threshold is "synthetic" (score >= threshold + 0.0).
    """
    prediction = [str(x) for x in baseline.predict(mfcc)]
    expected = np.where(scores >= baseline.threshold, SYNTHETIC, AUTHENTIC).tolist()
    _require_same_predictions("baseline", expected, prediction)
    return prediction


def _require_same_predictions(name: str, expected: list, actual: list) -> None:
    if len(expected) != len(actual):
        raise ValueError(f"{name}: {len(actual)} predictions for {len(expected)} scores")
    bad = [i for i, (e, a) in enumerate(zip(expected, actual)) if e != a]
    if bad:
        raise ValueError(f"{name}: {len(bad)} predictions differ from the independent recomputation, "
                         f"first at row {bad[0]}: {actual[bad[0]]!r} != {expected[bad[0]]!r}")


# --- Pre-claim checks (shared by run and rehearse) ------------------------------------------------

@dataclass
class Context:
    commit: str
    evidence: Evidence
    profile: Any
    manifest: Any
    models: Models
    checks: dict
    profile_config_sha256: str
    f2_manifest_sha256: str


def _load_models(checkpoints, baseline_dir, risk_policy_path, evidence: Evidence, batch_size: int,
                 checks: dict) -> Models:
    cnn, hashes = {}, {}
    for seed in SEEDS:
        expected, folder = evidence.checkpoint_sha256[seed], Path(checkpoints) / f"seed_{seed}"
        # The repository's hash (cnn_checkpoint._sha256_lines over both files, as load_checkpoint
        # computes it); load_checkpoint then checks the bytes it loads against expected again.
        _check(checks, f"checkpoint.seed_{seed}.sha256", expected, checkpoint_sha256(folder))
        loaded = load_checkpoint(folder, expected_sha256=expected)
        _check(checks, f"checkpoint.seed_{seed}.metadata_seed", seed, loaded[3]["seed"])
        cnn[seed], hashes[CNN_IDS[seed]] = loaded, expected

    _check(checks, "baseline.artifact_sha256", evidence.baseline_artifact_sha256,
           artifact_sha256(baseline_dir))
    baseline = load_baseline(baseline_dir)
    _check(checks, "baseline.version", BASELINE_ID, baseline.version)
    _check(checks, "baseline.threshold", evidence.baseline_threshold, baseline.threshold)
    hashes[BASELINE_ID] = evidence.baseline_artifact_sha256

    data = Path(risk_policy_path).read_bytes()  # one read: hashed bytes are parsed bytes
    _check(checks, "risk_policy.sha256", evidence.risk_policy_sha256, hashlib.sha256(data).hexdigest())
    policy = RiskPolicy.from_json(data.decode("utf-8"))
    return Models(cnn=cnn, baseline=baseline, policy=policy, bundle_seed=evidence.bundle_seed,
                  batch_size=batch_size, artifact_sha256=hashes)


def _load_cache(cache_dir, feature: str, split: str, profile, profile_config_sha256: str):
    features, labels, paths, meta = CACHE_LOADERS[feature](
        Path(cache_dir) / feature / split, split, profile, expected_f2_manifest_sha256=F2_MANIFEST_SHA256
    )
    if meta["profile_config_sha256"] != profile_config_sha256:
        raise ValueError(f"{feature}/{split}: cache built from profile config "
                         f"{meta['profile_config_sha256']}, profile file is {profile_config_sha256}")
    return features, labels, paths, meta


def reproduction_gate(cache_a, cache_b, profile, profile_config_sha256: str, evidence: Evidence,
                      checks: dict) -> None:
    """D46 (4): two val-A builds per feature, byte-identical, npz_sha256 equal to Stage 8 / Stage 7."""
    if Path(cache_a).resolve() == Path(cache_b).resolve():
        raise ValueError(f"cache_a and cache_b are the same folder ({cache_a}); the gate needs two builds")
    for feature in FEATURE_SHAPES:
        metas = [_load_cache(c, feature, GATE_SPLIT, profile, profile_config_sha256)[3]
                 for c in (cache_a, cache_b)]
        _check(checks, f"reproduction.{feature}.{GATE_SPLIT}.builds_identical",
               metas[0]["npz_sha256"], metas[1]["npz_sha256"])
        _check(checks, f"reproduction.{feature}.{GATE_SPLIT}.npz_sha256",
               evidence.cache_npz_sha256[(feature, GATE_SPLIT)], metas[0]["npz_sha256"])


def preflight(checks: dict, *, checkpoints, baseline_dir, risk_policy, profile_path, config_path,
              manifest_path, stage7_evidence, stage8_evidence, stage10_evidence, cache_a,
              cache_b) -> Context:
    """Every check that reads no test audio and writes nothing; raise on the first failure.

    Checks are recorded into the caller's dict as they run, so the caller keeps those done before
    a failure (the rehearsal report).
    """
    commit = code_version()["git_commit"]  # refuses a dirty worktree
    _record(checks, "code_commit", "full 40-hex SHA", commit,
            isinstance(commit, str) and _COMMIT.fullmatch(commit) is not None)
    environment_gate(checks)
    evidence = load_evidence(stage7_evidence, stage8_evidence, stage10_evidence)

    profile_path, config_path, manifest_path = Path(profile_path), Path(config_path), Path(manifest_path)
    profile = load_profile(profile_path)
    profile_config_sha256 = sha256_file(profile_path)
    f2_manifest_sha256 = sha256_file(manifest_path)
    _check(checks, "profile_fingerprint", evidence.profile_fingerprint, profile.fingerprint())
    _check(checks, "profile_config_sha256", evidence.profile_config_sha256, profile_config_sha256)
    _check(checks, "f2_manifest_sha256", evidence.f2_manifest_sha256, f2_manifest_sha256)
    _check(checks, "f2_manifest_sha256_approved", F2_MANIFEST_SHA256, f2_manifest_sha256)
    manifest = load_f2_manifest(manifest_path)
    _check(checks, "cnn_config_sha256", evidence.cnn_config_sha256, sha256_file(config_path))
    batch_size = load_cnn_config(config_path).batch_size
    _check(checks, "batch_size", evidence.batch_size, batch_size)

    models = _load_models(checkpoints, baseline_dir, risk_policy, evidence, batch_size, checks)
    reproduction_gate(cache_a, cache_b, profile, profile_config_sha256, evidence, checks)
    return Context(commit=commit, evidence=evidence, profile=profile, manifest=manifest, models=models,
                   checks=checks, profile_config_sha256=profile_config_sha256,
                   f2_manifest_sha256=f2_manifest_sha256)


def _model_set(models: Models) -> dict[str, dict]:
    def role(model_id):
        if model_id == BASELINE_ID:
            return ROLE_BASELINE
        return ROLE_BUNDLE if model_id == CNN_IDS[models.bundle_seed] else ROLE_OTHER
    return {m: {"role": role(m), "artifact_sha256": models.artifact_sha256[m]} for m in MODEL_IDS}


def _counts(rows) -> dict[str, dict[str, int]]:
    return {column: {k: int(v) for k, v in sorted(rows[column].value_counts().items())}
            for column in ("label", "subgroup")}


# --- Official run, phase 1: claim ---------------------------------------------------------------

def _check_results_ref(results_ref) -> str:
    """A non-empty, safe, canonical relative POSIX path (safe_relative_path), else ValueError."""
    if not isinstance(results_ref, str):
        raise ValueError(f"results_ref must be a string, got {results_ref!r}")
    try:
        canonical = safe_relative_path(results_ref)
    except ValueError as error:
        raise ValueError(f"results_ref must be a safe relative path: {error}") from None
    if canonical != results_ref:
        raise ValueError(f"results_ref must be the canonical POSIX path {canonical!r}, "
                         f"got {results_ref!r}")
    return results_ref


def _stage11_records(registry: RunRegistry) -> list:
    """Every (for, test, stage11_test) record, whatever its model set."""
    return [r for r in registry.records()
            if (r.dataset, r.split, r.evaluation_purpose) == (DATASET, SPLIT, PURPOSE)]


def claim(*, registry_path, out_dir, results_ref, **inputs) -> dict[str, Any]:
    """Phase 1 (module docstring, steps 1-3): check, claim, write the evidence; return claim.json.

    inputs are preflight()'s keyword arguments. No test audio is read. out_dir must not exist.
    """
    results_ref = _check_results_ref(results_ref)
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError(f"output folder already exists: {out_dir}")
    ctx = preflight({}, **inputs)
    counts = _counts(split_rows(ctx.manifest, SPLIT))  # manifest metadata only; no audio is read
    registry = RunRegistry(registry_path)  # an invalid or edited file fails here
    earlier = _stage11_records(registry)
    ctx.checks[f"registry.{DATASET}.{SPLIT}.{PURPOSE}.records"] = {
        "expected": 0, "actual": len(earlier), "passed": not earlier}
    if earlier:
        raise ValueError(f"the registry already holds a {earlier[0].status} {PURPOSE} record "
                         f"(run_id {earlier[0].run_id}); a second run needs a new decision (D31, D46 (8))")
    environment, freeze = environment_record()  # recorded, not asserted

    run_id = registry.claim(DATASET, SPLIT, MODEL_IDS, ctx.commit, evaluation_purpose=PURPOSE)
    last = RunRegistry(registry_path).records()[-1]  # re-read the claim from the file
    if (last.run_id, last.status, last.code_commit) != (run_id, STARTED, ctx.commit) \
            or last.key != (DATASET, SPLIT, PURPOSE, tuple(sorted(MODEL_IDS))):
        raise ValueError(f"the registry's last record is not this run's claim {run_id}: {last}")

    # From here on, a failure raises and the started record stays (D46 (8)).
    out_dir.mkdir(parents=True)  # exclusive: raises if the folder appeared since the first check
    environment_bytes = _json_text(environment).encode("utf-8")
    freeze_bytes = freeze.encode("utf-8")
    _create(out_dir / ENVIRONMENT_JSON, environment_bytes)
    _create(out_dir / PIP_FREEZE, freeze_bytes)
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "code_commit": ctx.commit,
        "registry_sha256": sha256_file(Path(registry_path)),
        "results_ref": results_ref,
        "dataset": DATASET,
        "split": SPLIT,
        "evaluation_purpose": PURPOSE,
        "model_ids": sorted(MODEL_IDS),
        "pre_claim_checks": ctx.checks,
        "environment_files": {ENVIRONMENT_JSON: hashlib.sha256(environment_bytes).hexdigest(),
                              PIP_FREEZE: hashlib.sha256(freeze_bytes).hexdigest()},
        "counts": counts,
    }
    _create(out_dir / CLAIM_JSON, _json_text(record).encode("utf-8"))
    return record


# --- Official run, phase 2: score -----------------------------------------------------------------

def _read_claim(out_dir: Path, run_id: str, commit: str, registry_path) -> tuple[dict, str]:
    """claim.json, checked against run_id, commit, the registry file and the environment files."""
    data = (out_dir / CLAIM_JSON).read_bytes()
    record = json.loads(data.decode("utf-8"))
    if (record.get("run_id"), record.get("code_commit")) != (run_id, commit):
        raise ValueError(f"{CLAIM_JSON} is for run_id {record.get('run_id')!r} at commit "
                         f"{record.get('code_commit')!r}, not {run_id} at {commit}")
    _check_results_ref(record.get("results_ref"))
    registry_sha256 = sha256_file(Path(registry_path))
    if registry_sha256 != record.get("registry_sha256"):
        raise ValueError(f"the registry changed since the claim: SHA-256 {registry_sha256}, "
                         f"{CLAIM_JSON} records {record.get('registry_sha256')}")
    for name, expected in record.get("environment_files", {}).items():
        if sha256_file(out_dir / name) != expected:
            raise ValueError(f"{name} changed since the claim")
    if set(record.get("environment_files", {})) != {ENVIRONMENT_JSON, PIP_FREEZE}:
        raise ValueError(f"{CLAIM_JSON} must record {ENVIRONMENT_JSON} and {PIP_FREEZE}")
    return record, hashlib.sha256(data).hexdigest()


def score(*, root, registry_path, out_dir, run_id, **inputs) -> dict[str, Any]:
    """Phase 2 (module docstring, steps 4-9): score the claimed run once; return run_report.json.

    out_dir is the folder claim() wrote. Before the marker, only claim.json and git are read, so a
    wrong run_id or commit is refused without writing anything. The marker is then created before
    anything else is read, so this function scores at most once per claim, whatever happens after.
    """
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):  # argument syntax only, no I/O
        raise ValueError(f"run_id must be 32 lowercase hex chars, got {run_id!r}")
    out_dir = Path(out_dir)
    claim_path = out_dir / CLAIM_JSON
    if not claim_path.is_file():
        raise ValueError(f"{claim_path} does not exist: score needs the folder written by claim")
    claimed = json.loads(claim_path.read_bytes().decode("utf-8"))  # local evidence, no test access
    commit = code_version()["git_commit"]
    if (claimed.get("run_id"), claimed.get("code_commit")) != (run_id, commit):
        raise ValueError(f"{CLAIM_JSON} is for run_id {claimed.get('run_id')!r} at commit "
                         f"{claimed.get('code_commit')!r}, not {run_id} at {commit}; nothing was written")

    marker = {"run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        _create(out_dir / SCORING_MARKER, _json_text(marker).encode("utf-8"))
    except ValueError:
        raise ValueError(f"{out_dir / SCORING_MARKER} exists: scoring was already started for this claim; "
                         "no rerun after the claim without a new decision (D46 (8))") from None
    # The scoring session's environment, recorded, not asserted; compared with the claim's below.
    environment_score_bytes = _json_text(environment_record()[0]).encode("utf-8")
    _create(out_dir / ENVIRONMENT_SCORE_JSON, environment_score_bytes)

    ctx = preflight({}, **inputs)
    rows = split_rows(ctx.manifest, SPLIT)
    records = _stage11_records(RunRegistry(registry_path))
    if len(records) != 1 or (records[0].run_id, records[0].status, records[0].code_commit) \
            != (run_id, STARTED, ctx.commit) or records[0].model_set != tuple(sorted(MODEL_IDS)):
        found = [(r.run_id, r.status, r.code_commit) for r in records]
        raise ValueError(f"the registry must hold exactly one started {PURPOSE} record for run_id "
                         f"{run_id} at commit {ctx.commit}, found {found}")
    claim_record, claim_sha256 = _read_claim(out_dir, run_id, ctx.commit, registry_path)
    if _counts(rows) != claim_record["counts"]:
        raise ValueError(f"test rows changed since the claim: {_counts(rows)} != {claim_record['counts']}")

    features, feature_hashes = compute_features(root, rows, ctx.profile)
    models = ctx.models
    scores = score_models(models, features, MODEL_IDS)
    bundle_id = CNN_IDS[models.bundle_seed]
    p, bundle_prediction = bundle_outputs(models.policy, scores[bundle_id])
    baseline_prediction = baseline_outputs(models.baseline, features[MFCC], scores[BASELINE_ID])

    model_set = _model_set(models)
    clip_dir = out_dir / CLIP_SCORES_DIR
    clip_dir.mkdir()  # exclusive
    columns = {"clip_id": rows["path"].tolist(), "label": rows["label"].tolist(),
               "group_id": rows["group"].tolist()}
    outputs = {BASELINE_ID: {"prediction": baseline_prediction},
               bundle_id: {"p": p, "prediction": bundle_prediction}}
    clip_files = {}
    for model_id in MODEL_IDS:
        csv_path, json_path = write_clip_scores(
            clip_dir, evaluation_purpose=PURPOSE, run_id=run_id, model_id=model_id,
            model_role=model_set[model_id]["role"],
            model_artifact_sha256=model_set[model_id]["artifact_sha256"],
            manifest_sha256=ctx.f2_manifest_sha256, raw_score=scores[model_id], **columns,
            **outputs.get(model_id, {}),
        )
        clip_files[model_id] = {"csv_sha256": sha256_file(csv_path), "sidecar_sha256": sha256_file(json_path),
                                "n_rows": len(rows)}

    report = {
        "schema_version": SCHEMA_VERSION,
        "code_commit": ctx.commit,
        "run_id": run_id,
        "dataset": DATASET,
        "split": SPLIT,
        "evaluation_purpose": PURPOSE,
        "model_set": model_set,
        "bundle_seed": models.bundle_seed,
        "claim_sha256": claim_sha256,
        "scoring_marker_sha256": sha256_file(out_dir / SCORING_MARKER),
        "pre_scoring_checks": ctx.checks,
        "environment_files": claim_record["environment_files"],
        "environment_score_sha256": hashlib.sha256(environment_score_bytes).hexdigest(),
        # Descriptive only: the two canonical records byte-equal (claim session vs scoring session).
        "environment_same_as_claim": environment_score_bytes == (out_dir / ENVIRONMENT_JSON).read_bytes(),
        "scoring": {"device": DEVICE, "batch_size": models.batch_size, "passes": 2},
        "features": feature_hashes,
        "clip_scores": {"folder": CLIP_SCORES_DIR, "files": clip_files},
        "counts": claim_record["counts"],
    }
    _create(out_dir / RUN_REPORT, _json_text(report).encode("utf-8"))
    RunRegistry(registry_path).complete(run_id, results_ref=claim_record["results_ref"])
    return report


# --- Dry run (validation splits only) -------------------------------------------------------------

def _check_rehearsal_splits(splits) -> tuple[str, ...]:
    if isinstance(splits, str):
        raise ValueError(f"splits must be a sequence of split names, not the string {splits!r}")
    splits = tuple(splits)
    for split in splits:
        if not isinstance(split, str) or split not in REHEARSAL_SPLITS:
            raise ValueError(f"the rehearsal reads {list(REHEARSAL_SPLITS)} only, got {split!r}")
    if sorted(splits) != sorted(REHEARSAL_SPLITS):
        raise ValueError(f"the rehearsal needs exactly {list(REHEARSAL_SPLITS)}, got {list(splits)}")
    return splits


def _load_val_b_scores(path, expected_sha256: str, checks: dict) -> dict[str, np.ndarray]:
    data = Path(path).read_bytes()  # one read: hashed bytes are parsed bytes
    _check(checks, "stage10.val_b_scores_sha256", expected_sha256, hashlib.sha256(data).hexdigest())
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            arrays = {key: npz[key] for key in npz.files}
    except (zipfile.BadZipFile, OSError, EOFError, ValueError) as error:
        raise ValueError(f"{path} is not a readable non-pickle .npz: {error}") from error
    if set(arrays) != _VAL_B_SCORE_KEYS:
        raise ValueError(f"{path}: entries {sorted(arrays)}, expected {sorted(_VAL_B_SCORE_KEYS)}")
    return arrays


def _bytes_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _write_rehearsal_report(out_dir: Path, report: dict, freeze: str) -> None:
    out_dir.mkdir(parents=True)
    freeze_bytes = freeze.encode("utf-8")
    _create(out_dir / PIP_FREEZE, freeze_bytes)
    report["environment"]["pip_freeze_file_sha256"] = hashlib.sha256(freeze_bytes).hexdigest()
    _create(out_dir / REHEARSAL_REPORT, _json_text(report).encode("utf-8"))


def rehearse(*, root, stage10_scores, out_dir, splits=REHEARSAL_SPLITS, **inputs) -> dict[str, Any]:
    """The D46 dry run on val-A and val-B (module docstring); return the rehearsal report.

    cache_a must also hold the val-B caches. No registry is opened and no test path exists here.
    After the split and folder checks, the report is always written: "passed": true on success;
    on any failure "passed": false, the failing check's name (None if the error was not a recorded
    check) and the error, and the error is raised again.
    """
    splits = _check_rehearsal_splits(splits)  # before anything is read
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError(f"output folder already exists: {out_dir}")
    environment, freeze = environment_record()
    checks: dict = {}
    known: dict = {}
    per_split: dict = {}
    report = {
        "schema_version": SCHEMA_VERSION,
        "code_commit": None,
        "splits": list(splits),
        "pre_claim_checks": checks,
        "known_answers": known,
        "per_split": per_split,
        "environment": {**environment, "pip_freeze_file": PIP_FREEZE},
    }
    try:
        ctx = preflight(checks, **inputs)
        models, evidence = ctx.models, ctx.evidence
        report.update({"code_commit": ctx.commit, "model_set": _model_set(models),
                       "bundle_seed": models.bundle_seed,
                       "scoring": {"device": DEVICE, "batch_size": models.batch_size, "passes": 2}})
        bundle_id = CNN_IDS[models.bundle_seed]
        for split in splits:
            rows = split_rows(ctx.manifest, split)
            features, feature_hashes = compute_features(root, rows, ctx.profile)
            for feature in FEATURE_SHAPES:  # in-memory features byte-equal to the verified cache
                cached, labels, paths, meta = _load_cache(inputs["cache_a"], feature, split, ctx.profile,
                                                          ctx.profile_config_sha256)
                _check(known, f"{split}.{feature}.cache_npz_sha256",
                       evidence.cache_npz_sha256[(feature, split)], meta["npz_sha256"])
                cache_hash = features_sha256(paths.tolist(), [LABEL_NAMES[int(x)] for x in labels],
                                             {feature: array_sha256(feature, cached)})
                memory_hash = features_sha256(rows["path"].tolist(), rows["label"].tolist(),
                                              {feature: feature_hashes["arrays"][feature]})
                _check(known, f"{split}.{feature}.in_memory_equals_cache", cache_hash, memory_hash)
            labels = rows["label"].map(LABEL_CODES).to_numpy()

            if split == "val-A":
                scores = score_models(models, features, MODEL_IDS)
                for seed in SEEDS:
                    _check(known, f"val-A.{CNN_IDS[seed]}.eer", models.cnn[seed][3]["val_a_eer"],
                           eer(labels, scores[CNN_IDS[seed]])[0])
                _check(known, f"val-A.{BASELINE_ID}.eer", evidence.baseline_val_a_eer,
                       eer(labels, scores[BASELINE_ID])[0])
            else:  # val-B: the bundle seed (D45) and the baseline only
                scores = score_models(models, features, (BASELINE_ID, bundle_id))
                stage10 = _load_val_b_scores(stage10_scores, evidence.val_b_scores_sha256, known)
                p, _ = bundle_outputs(models.policy, scores[bundle_id])
                _check(known, "val-B.stage10.paths", rows["path"].tolist(), stage10["paths"].tolist())
                _check(known, "val-B.stage10.labels", labels.tolist(),
                       stage10["labels"].astype(int).tolist())
                _check(known, f"val-B.{bundle_id}.raw_score_bytes", _bytes_sha256(stage10["score"]),
                       _bytes_sha256(scores[bundle_id]))
                _check(known, f"val-B.{bundle_id}.p_bytes", _bytes_sha256(stage10["p"]),
                       _bytes_sha256(p))
                _check(known, f"val-B.{BASELINE_ID}.eer", evidence.baseline_val_b_eer,
                       eer(labels, scores[BASELINE_ID])[0])
                baseline_outputs(models.baseline, features[MFCC], scores[BASELINE_ID])
            per_split[split] = {"features": feature_hashes, "scored": sorted(scores),
                                "counts": _counts(rows)}
    except Exception as error:
        if report["code_commit"] is None and "code_commit" in checks:  # preflight failed after reading it
            report["code_commit"] = checks["code_commit"]["actual"]
        report.update({"passed": False, "failed_check": getattr(error, "key", None),
                       "error": f"{type(error).__name__}: {error}"})
        _write_rehearsal_report(out_dir, report, freeze)
        raise
    # Never "passed": true unless every recorded check passed.
    passed = all(c["passed"] for c in (*checks.values(), *known.values()))
    report.update({"passed": passed, "failed_check": None, "error": None})
    _write_rehearsal_report(out_dir, report, freeze)
    if not passed:
        raise ValueError("a rehearsal check is recorded as failed")
    return report


# --- CLI ------------------------------------------------------------------------------------------

def _inputs(args) -> dict[str, Any]:
    return {
        "checkpoints": args.checkpoints, "baseline_dir": args.baseline, "risk_policy": args.risk_policy,
        "profile_path": args.profile, "config_path": args.config, "manifest_path": args.manifest,
        "stage7_evidence": args.stage7_evidence, "stage8_evidence": args.stage8_evidence,
        "stage10_evidence": args.stage10_evidence, "cache_a": args.cache_a, "cache_b": args.cache_b,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("rehearse", "dry run on val-A and val-B (no registry)"),
                       ("claim", "official run, phase 1: checks, registry claim, evidence (no audio)"),
                       ("score", "official run, phase 2: score the claimed run once")):
        p = commands.add_parser(name, help=text)
        if name != "claim":
            p.add_argument("--root", required=True, type=Path, help="Local extracted FoR root")
        p.add_argument("--checkpoints", required=True, type=Path,
                       help="Folder holding seed_0/ seed_1/ seed_2/")
        p.add_argument("--baseline", required=True, type=Path, help="baseline-v1 artifact folder")
        p.add_argument("--risk-policy", required=True, type=Path, help="risk_policy.json (Stage 10)")
        p.add_argument("--profile", required=True, type=Path, help="F3 preprocessing profile YAML")
        p.add_argument("--config", required=True, type=Path, help="configs/cnn.yaml (batch_size)")
        p.add_argument("--manifest", required=True, type=Path, help="Approved F2 FoR manifest")
        p.add_argument("--stage7-evidence", required=True, type=Path,
                       help="docs/audit_evidence/stage7_baseline")
        p.add_argument("--stage8-evidence", required=True, type=Path, help="docs/audit_evidence/stage8_cnn")
        p.add_argument("--stage10-evidence", required=True, type=Path,
                       help="docs/audit_evidence/stage10_risk")
        p.add_argument("--cache-a", required=True, type=Path, help="Build A: log_mel/<split>/, mfcc/<split>/")
        p.add_argument("--cache-b", required=True, type=Path, help="Build B: log_mel/val-A/, mfcc/val-A/")
        p.add_argument("--out", required=True, type=Path,
                       help="rehearse, claim: new output folder; score: the folder claim wrote")
        if name == "rehearse":
            p.add_argument("--stage10-scores", required=True, type=Path, help="Stage 10 val_b_scores.npz")
        else:
            p.add_argument("--registry", required=True, type=Path, help="Run registry JSON Lines file")
        if name == "claim":
            p.add_argument("--results-ref", required=True,
                           help="Relative POSIX path complete() records, e.g. stage11/run_report.json")
        if name == "score":
            p.add_argument("--run-id", required=True, help="The run_id printed by claim")
    args = parser.parse_args(argv)
    if args.command == "claim":
        report = claim(registry_path=args.registry, out_dir=args.out, results_ref=args.results_ref,
                       **_inputs(args))
    elif args.command == "score":
        report = score(root=args.root, registry_path=args.registry, out_dir=args.out, run_id=args.run_id,
                       **_inputs(args))
    else:
        report = rehearse(root=args.root, stage10_scores=args.stage10_scores, out_dir=args.out,
                          **_inputs(args))
    print(_json_text(report), end="")


if __name__ == "__main__":
    main()
