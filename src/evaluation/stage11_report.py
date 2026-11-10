"""Stage 11 metrics report, computed only from the files the Stage 11 run stored (Protocol v1.0 §7).

Reads the output folder of src.evaluation.stage11_run score (run_report.json, claim.json and the four
per-clip score files), the run registry, the F2 manifest, the frozen risk policy and the committed
Stage 7 and Stage 10 evidence. It reads no audio, no feature cache, no checkpoint and no model, and
never calls the runner, so the metrics can be recomputed and reviewed without a second test run
(D31; docs/stage11_run.md Why (g)).

Order:
1. Integrity checks, each recorded as expected / actual / passed and raising ValueError naming the
   check: run report, claim, registry (one started and one completed record), each score file
   against its sidecar and the run report, the same clips in all four files, the F2 join (split,
   label, group, no clip missing, counts), the frozen thresholds and hashes, and an independent
   re-derivation of every stored p and prediction.
2. Metrics under the D57 reporting conventions (CONVENTIONS below): per model score level (EER,
   ROC-AUC) on raw_score, overall and REAL vs each FAKE subgroup; forced-binary metrics for the
   bundle-seed CNN (on p at the frozen threshold) and the baseline (on raw_score at its D29
   threshold), overall and per subgroup; three-way metrics for the bundle-seed CNN from the stored
   predictions; group-clustered bootstrap CIs (D11, D14); H1, H5; the seed rows (Protocol §5).
3. write_report: metrics_report.json (strict JSON) and metrics_report.md, rendered from the JSON
   only, into a new folder.

The split is read from run_report.json and must equal claim.json and the registry record; this
module never names it. Torch-free: the runner's constants it needs are re-declared here, and a test
checks they equal the runner's. Nothing is logged to W&B and nothing is written outside --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.clip_scores import read_clip_scores
from src.evaluation.metrics import binary_metrics, eer, roc_auc, three_way_metrics
from src.evaluation.run_registry import COMPLETED, STARTED, RunRegistry
from src.evaluation.seeds import seed_summary
from src.features.mfcc_cache import LABEL_CODES
from src.preprocessing.validation import F2_COLUMNS, F2_MANIFEST_SHA256, _safe_relative, code_version
from src.risk.decisions import AUTHENTIC, DECISIONS, SYNTHETIC, UNCERTAIN
from src.risk.policy import RiskPolicy
from src.tuning.baseline_run import FAKE_SUBGROUPS, _json_text, _subgroups

# Re-declared from src.evaluation.stage11_run (which imports torch); a test checks equality.
RUN_SCHEMA_VERSION = 1
DATASET = "for"
PURPOSE = "stage11_test"
SEEDS = (0, 1, 2)
BASELINE_ID = "baseline-v1"
CNN_IDS = {seed: f"cnn-v1-seed{seed}" for seed in SEEDS}
MODEL_IDS = (BASELINE_ID, *CNN_IDS.values())
BASELINE_ARTIFACT_SHA256 = "42cc4d53a7cd881f56e4fa54fcf4797de8885b404d0a2b31c293555c868df10a"  # D41
ROLE_BASELINE = "baseline"
ROLE_BUNDLE = "cnn_bundle_seed"
ROLE_OTHER = "cnn_other_seed"
CLIP_SCORES_DIR = "clip_scores"
RUN_REPORT = "run_report.json"
CLAIM_JSON = "claim.json"
STAGE_REPORT = "run_report.json"  # the Stage 7 and Stage 10 evidence file name

SCHEMA_VERSION = 1
REPORT_JSON = "metrics_report.json"
REPORT_MD = "metrics_report.md"
SUBGROUPS = ("REAL", *FAKE_SUBGROUPS)
APPROVED_F2_MANIFEST_SHA256 = F2_MANIFEST_SHA256
H5_MIN_DIFFERENCE = Fraction(5, 100)  # Protocol §7: "≥ 5 points"
_HEX32 = re.compile(r"[0-9a-f]{32}")
_HEX40 = re.compile(r"[0-9a-f]{40}")

# D57: reporting conventions, fixed before any FoR test result (docs/decisions/README.md).
CONVENTIONS = {
    "1_cnn": "CNN in H1 and H5 = the bundle-seed CNN (role cnn_bundle_seed, seed from run_report.json); "
             "seeds 1-2 at score level only (Protocol §5, D45)",
    "2_score_level": "EER and ROC-AUC on raw_score for every model, never on p",
    "3_subgroups": "score level: REAL vs FAKE_mp3 and REAL vs FAKE_other; decision level: each subgroup "
                   "alone (binary_metrics / three_way_metrics on the subset)",
    "4_decision_level": "bundle CNN: forced binary = binary_metrics(labels, p, policy.threshold), three-way "
                        "= three_way_metrics(labels, stored prediction); baseline: forced binary only, "
                        "binary_metrics(labels, raw_score, Stage 7 val_b_threshold) (D4, D29)",
    "5_h5": "H5 recall = forced-binary recall of the bundle CNN at the frozen threshold; d = recall(FAKE_mp3) "
            "- recall(FAKE_other); clearly lower iff d >= 0.05 (compared exactly on the counts) and the CI "
            "lower bound > 0",
    "6_h1": "H1 d = EER(baseline-v1) - EER(bundle CNN), paired on the same resampled indices; supported iff "
            "d > 0 and the CI lower bound > 0",
    "7_ci": "bootstrap_ci defaults (95% percentile, 2,000 resamples, seed 2026), clusters = group_id (D14), "
            "on all rows; for the bundle CNN and the baseline: overall EER, ROC-AUC, forced-binary FPR and "
            "FNR; H1 d; H5 d; counts, n_invalid and collapsed (low == high) reported; subgroup rows: point "
            "estimates with counts, no CIs",
    "8_seeds": "seed_summary of test EER and ROC-AUC over seeds 0, 1, 2 (mean, sample SD, values), a "
               "separate row next to the bundle-seed CIs",
    "9_undefined": "undefined rates (denominator 0) are JSON null; strict JSON; the EER threshold is "
                   "descriptive only; realised FPR/FNR are at the frozen thresholds",
}
H1_RULE = ("H1 supported iff d = EER(baseline-v1) - EER(bundle CNN) > 0 and the lower bound of its 95% "
           "paired cluster-bootstrap CI > 0 (Protocol §1; D57 (6))")
H5_RULE = ("H5 'clearly lower' iff d = recall(FAKE_mp3) - recall(FAKE_other) of the bundle CNN, forced "
           "binary at the frozen threshold, is >= 0.05 and the lower bound of its 95% cluster-bootstrap "
           "CI > 0 (Protocol §7; D57 (5))")


class CheckFailed(ValueError):
    """An integrity check failed; key names it."""

    def __init__(self, key: str, message: str):
        super().__init__(message)
        self.key = key


def _record(checks: dict, key: str, expected, actual, passed: bool) -> None:
    checks[key] = {"expected": expected, "actual": actual, "passed": bool(passed)}
    if not passed:
        raise CheckFailed(key, f"check failed: {key}: expected {expected!r}, got {actual!r}")


def _check(checks: dict, key: str, expected, actual) -> None:
    """_record with passed = (actual == expected); a bool never equals a number."""
    same = isinstance(actual, bool) == isinstance(expected, bool) and actual == expected
    _record(checks, key, expected, actual, bool(same))


def _fail(checks: dict, key: str, error: Exception) -> CheckFailed:
    checks[key] = {"expected": "no error", "actual": f"{type(error).__name__}: {error}", "passed": False}
    return CheckFailed(key, f"check failed: {key}: {error}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> tuple[dict, bytes]:
    data = Path(path).read_bytes()
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return value, data


def _digest(values) -> str:
    """SHA-256 of the values joined by newlines (a short, comparable record of a long column)."""
    return _sha256("\n".join(map(str, values)).encode("utf-8"))


# --- 3a. Integrity checks -------------------------------------------------------------------------

def _check_run_report(checks: dict, report: dict) -> int:
    """Check 1: run_report.json; return the bundle seed."""
    _check(checks, "run_report.schema_version", RUN_SCHEMA_VERSION, report.get("schema_version"))
    _check(checks, "run_report.dataset", DATASET, report.get("dataset"))
    _check(checks, "run_report.evaluation_purpose", PURPOSE, report.get("evaluation_purpose"))
    run_id, commit = report.get("run_id"), report.get("code_commit")
    _record(checks, "run_report.run_id", "32 lowercase hex", run_id,
            isinstance(run_id, str) and _HEX32.fullmatch(run_id) is not None)
    _record(checks, "run_report.code_commit", "40 lowercase hex", commit,
            isinstance(commit, str) and _HEX40.fullmatch(commit) is not None)
    split = report.get("split")
    _record(checks, "run_report.split", "non-empty string", split, isinstance(split, str) and bool(split))
    model_set = report.get("model_set")
    ids = sorted(model_set) if isinstance(model_set, dict) else None
    _check(checks, "run_report.model_set.ids", sorted(MODEL_IDS), ids)
    roles = sorted(model_set[m].get("role") for m in MODEL_IDS)
    _check(checks, "run_report.model_set.roles", sorted([ROLE_BASELINE, ROLE_BUNDLE, ROLE_OTHER, ROLE_OTHER]),
           roles)
    _check(checks, "run_report.model_set.baseline_role", ROLE_BASELINE, model_set[BASELINE_ID]["role"])
    seed = report.get("bundle_seed")
    _record(checks, "run_report.bundle_seed", f"one of {list(SEEDS)}", seed,
            not isinstance(seed, bool) and seed in SEEDS)
    _check(checks, "run_report.bundle_seed_role", ROLE_BUNDLE, model_set[CNN_IDS[seed]]["role"])
    return seed


def _check_claim(checks: dict, report: dict, claim: dict, claim_bytes: bytes) -> None:
    """Check 2: claim.json names the same run and its bytes are the ones the run report hashed."""
    for key in ("run_id", "code_commit", "split", "dataset", "evaluation_purpose"):
        _check(checks, f"claim.{key}", report[key], claim.get(key))
    _check(checks, "claim.sha256", report.get("claim_sha256"), _sha256(claim_bytes))
    ref = claim.get("results_ref")
    _record(checks, "claim.results_ref", "non-empty string", ref, isinstance(ref, str) and bool(ref.strip()))


def _check_registry(checks: dict, report: dict, claim: dict, registry_path: Path) -> None:
    """Check 3: exactly one started and one completed record for this run_id, matching the run."""
    try:
        records = RunRegistry(registry_path).records()
    except ValueError as error:
        raise _fail(checks, "registry.load", error) from None
    run_id = report["run_id"]
    mine = [r for r in records if r.run_id == run_id]
    statuses = sorted(r.status for r in mine)
    _check(checks, "registry.records", sorted([COMPLETED, STARTED]), statuses)
    expected_key = [report["dataset"], report["split"], report["evaluation_purpose"], sorted(MODEL_IDS)]
    for record in mine:
        _check(checks, f"registry.{record.status}.key", expected_key,
               [record.dataset, record.split, record.evaluation_purpose, list(record.model_set)])
        _check(checks, f"registry.{record.status}.code_commit", report["code_commit"], record.code_commit)
    completed = next(r for r in mine if r.status == COMPLETED)
    _check(checks, "registry.completed.results_ref", claim["results_ref"], completed.results_ref)


def _read_scores(checks: dict, report: dict, clip_dir: Path, manifest_sha256: str) -> dict:
    """Check 4: each score file read and verified, and bound to the run report and the F2 manifest."""
    files = report.get("clip_scores", {})
    _check(checks, "clip_scores.folder", CLIP_SCORES_DIR, files.get("folder"))
    _check(checks, "clip_scores.models", sorted(MODEL_IDS), sorted(files.get("files", {})))
    out = {}
    for model_id in MODEL_IDS:
        expected = files["files"][model_id]
        try:
            scores = read_clip_scores(clip_dir, model_id)
        except ValueError as error:
            raise _fail(checks, f"clip_scores.{model_id}.read", error) from None
        csv_bytes = (clip_dir / f"{model_id}.csv").read_bytes()
        sidecar_bytes = (clip_dir / f"{model_id}.json").read_bytes()
        _check(checks, f"clip_scores.{model_id}.csv_sha256", expected.get("csv_sha256"), _sha256(csv_bytes))
        _check(checks, f"clip_scores.{model_id}.sidecar_sha256", expected.get("sidecar_sha256"),
               _sha256(sidecar_bytes))
        _check(checks, f"clip_scores.{model_id}.n_rows", expected.get("n_rows"), len(scores.clip_id))
        meta, model = scores.metadata, report["model_set"][model_id]
        _check(checks, f"clip_scores.{model_id}.run_id", report["run_id"], meta["run_id"])
        _check(checks, f"clip_scores.{model_id}.evaluation_purpose", PURPOSE, meta["evaluation_purpose"])
        _check(checks, f"clip_scores.{model_id}.model_role", model["role"], meta["model_role"])
        _check(checks, f"clip_scores.{model_id}.model_artifact_sha256", model["artifact_sha256"],
               meta["model_artifact_sha256"])
        _check(checks, f"clip_scores.{model_id}.manifest_sha256", manifest_sha256, meta["manifest_sha256"])
        out[model_id] = scores
    return out


def _check_same_clips(checks: dict, scores: dict) -> None:
    """Check 5: the same clip IDs, labels and groups, in the same order, in all four files."""
    first = scores[BASELINE_ID]
    for name in ("clip_id", "label", "group_id"):
        expected = _digest(getattr(first, name))
        for model_id in MODEL_IDS[1:]:
            _check(checks, f"same_clips.{model_id}.{name}", expected, _digest(getattr(scores[model_id], name)))


def _load_manifest(checks: dict, path: Path, expected_sha256: str) -> tuple[pd.DataFrame, str]:
    """The F2 manifest, read as load_f2_manifest does (all columns str, canonical paths)."""
    data = path.read_bytes()
    sha = _sha256(data)
    _check(checks, "manifest.sha256_stage10", expected_sha256, sha)
    _check(checks, "manifest.sha256_approved", APPROVED_F2_MANIFEST_SHA256, sha)
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    _check(checks, "manifest.columns", list(F2_COLUMNS), list(manifest.columns))
    manifest["path"] = manifest["path"].map(_safe_relative)
    _check(checks, "manifest.unique_paths", 0, int(manifest["path"].duplicated().sum()))
    return manifest, sha


def _f2_join(checks: dict, report: dict, manifest: pd.DataFrame, clips) -> np.ndarray:
    """Check 6: every clip in F2 with the run's split, label and group; none missing; counts equal."""
    split = report["split"]
    paths = np.array(clips.clip_id, dtype=object)
    labels = np.array([LABEL_CODES[x] for x in clips.label])
    try:
        subgroups = _subgroups(manifest, split, paths, labels)  # path, split and label (Stage 7 join)
    except ValueError as error:
        raise _fail(checks, "f2.rows", error) from None
    checks["f2.rows"] = {"expected": "every clip in F2 with the run's split and label",
                         "actual": f"{len(paths)} clips found", "passed": True}
    groups = manifest.set_index("path").loc[list(paths), "group"].tolist()
    _check(checks, "f2.group", _digest(groups), _digest(clips.group_id))
    _check(checks, "f2.n_rows", int((manifest["split"] == split).sum()), len(paths))
    counts = {"label": {k: int(v) for k, v in sorted(pd.Series(list(clips.label)).value_counts().items())},
              "subgroup": {k: int(v) for k, v in sorted(pd.Series(subgroups).value_counts().items())}}
    _check(checks, "f2.counts", report.get("counts"), counts)
    return subgroups.astype(str)


def _check_thresholds(checks: dict, report: dict, policy_bytes: bytes, s7: dict, s10: dict, bundle_seed: int):
    """Check 7: the frozen policy and the baseline threshold, bound to Stage 7 and Stage 10."""
    _check(checks, "risk_policy.sha256", s10.get("risk_policy_sha256"), _sha256(policy_bytes))
    policy = RiskPolicy.from_json(policy_bytes.decode("utf-8"))
    _check(checks, "stage10.bundle_seed", s10.get("bundle_seed"), bundle_seed)
    _check(checks, "stage10.checkpoint_sha256", s10.get("checkpoint_sha256"),
           report["model_set"][CNN_IDS[bundle_seed]]["artifact_sha256"])
    _check(checks, "baseline.artifact_sha256.stage7", s7.get("artifact_sha256"),
           report["model_set"][BASELINE_ID]["artifact_sha256"])
    _check(checks, "baseline.artifact_sha256.d41", BASELINE_ARTIFACT_SHA256, s7.get("artifact_sha256"))
    threshold = s7.get("val_b_threshold")
    _record(checks, "baseline.threshold", "finite number from Stage 7 val_b_threshold", threshold,
            isinstance(threshold, float) and np.isfinite(threshold))
    return policy, threshold


def _rederive(checks: dict, policy: RiskPolicy, bundle, baseline, baseline_threshold: float) -> None:
    """Check 8: every stored p and prediction follows from the stored raw score (independent of the runner)."""
    p = policy.calibration.transform(bundle.raw_score)
    _check(checks, "rederive.bundle.p_bytes", _sha256(p.tobytes()), _sha256(bundle.p.tobytes()))
    # The band rule of src/risk/decisions.py three_way_decisions, written out:
    # synthetic if p >= threshold + half_width; authentic if p < threshold - half_width; else uncertain.
    upper, lower = policy.threshold + policy.half_width, policy.threshold - policy.half_width
    expected = np.where(bundle.p >= upper, SYNTHETIC, np.where(bundle.p < lower, AUTHENTIC, UNCERTAIN))
    _check(checks, "rederive.bundle.prediction_mismatches", 0,
           int(np.sum(expected != np.array(bundle.prediction, dtype=object))))
    expected = np.where(baseline.raw_score >= baseline_threshold, SYNTHETIC, AUTHENTIC)
    _check(checks, "rederive.baseline.prediction_mismatches", 0,
           int(np.sum(expected != np.array(baseline.prediction, dtype=object))))


# --- 3b. Metrics ----------------------------------------------------------------------------------

def _num(x) -> float | None:
    """A rate as JSON: NaN (undefined, denominator 0) -> None."""
    x = float(x)
    return None if np.isnan(x) else x


def _binary(m) -> dict[str, Any]:
    return {k: (int(v) if isinstance(v, (int, np.integer)) else _num(v)) for k, v in vars(m).items()}


def _three_way(m) -> dict[str, Any]:
    return {
        "confusion": {"columns": list(DECISIONS), "bonafide": list(m.confusion[0]), "spoof": list(m.confusion[1])},
        "n": m.n, "n_deferred": m.n_deferred, "coverage": _num(m.coverage), "deferral_rate": _num(m.deferral_rate),
        "deferral_rate_bonafide": _num(m.deferral_rate_bonafide), "deferral_rate_spoof": _num(m.deferral_rate_spoof),
        "non_deferred": _binary(m.non_deferred),
    }


def _score_level(y: np.ndarray, s: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    rate, threshold = eer(y[mask], s[mask])
    return {"eer": rate, "threshold_at_eer_point_descriptive": threshold, "roc_auc": roc_auc(y[mask], s[mask]),
            "n_spoof": int(y[mask].sum()), "n_bonafide": int((y[mask] == 0).sum())}


def _ci(statistic, n: int, groups, counts: dict) -> dict[str, Any]:
    ci = bootstrap_ci(statistic, n, clusters=groups)
    return {"estimate": ci.estimate, "low": ci.low, "high": ci.high, "confidence": ci.confidence,
            "n_resamples": ci.n_resamples, "n_invalid": ci.n_invalid, "collapsed": ci.low == ci.high,
            "n_groups": len(set(groups)), "counts": counts}


def _model_cis(y, raw, decision_score, threshold, groups) -> dict[str, Any]:
    """CIs of one model's overall EER and ROC-AUC (on raw, D57 (2)) and forced-binary FPR / FNR
    (decision_score >= threshold: p for the bundle CNN, raw for the baseline; D57 (4), (7))."""
    n = len(y)
    fb = binary_metrics(y, decision_score, threshold)
    n_counts = {"n_spoof": fb.n_spoof, "n_bonafide": fb.n_bonafide}
    return {
        "eer": _ci(lambda i: eer(y[i], raw[i])[0], n, groups, n_counts),
        "roc_auc": _ci(lambda i: roc_auc(y[i], raw[i]), n, groups, n_counts),
        "fpr": _ci(lambda i: binary_metrics(y[i], decision_score[i], threshold).fpr, n, groups,
                   {"fp": fb.fp, "n_bonafide": fb.n_bonafide}),
        "fnr": _ci(lambda i: binary_metrics(y[i], decision_score[i], threshold).fnr, n, groups,
                   {"fn": fb.fn, "n_spoof": fb.n_spoof}),
    }


def _subgroup_rows(y, subgroups, fn) -> dict[str, Any]:
    return {name: fn(subgroups == name) for name in SUBGROUPS}


def _models(y, subgroups, scores, policy, baseline_threshold, groups, bundle_id) -> dict[str, Any]:
    out = {}
    for model_id in MODEL_IDS:
        clip = scores[model_id]
        s = clip.raw_score
        everything = np.ones(len(y), dtype=bool)
        entry: dict[str, Any] = {
            "role": clip.metadata["model_role"],
            "score_level": {
                "overall": _score_level(y, s, everything),
                **{f"REAL_vs_{f}": _score_level(y, s, (subgroups == "REAL") | (subgroups == f))
                   for f in FAKE_SUBGROUPS},
            },
        }
        if model_id == bundle_id:
            p, t = clip.p, policy.threshold
            stored = np.array(clip.prediction, dtype=object)
            entry["frozen"] = {"threshold_on_p": t, "half_width": policy.half_width, "policy_version": policy.version}
            entry["forced_binary"] = {"overall": _binary(binary_metrics(y, p, t)),
                                      **_subgroup_rows(y, subgroups, lambda m: _binary(binary_metrics(y[m], p[m], t)))}
            entry["three_way"] = {"overall": _three_way(three_way_metrics(y, stored)),
                                  **_subgroup_rows(y, subgroups,
                                                   lambda m: _three_way(three_way_metrics(y[m], stored[m])))}
            entry["ci"] = _model_cis(y, s, p, t, groups)
        elif model_id == BASELINE_ID:
            t = baseline_threshold
            entry["frozen"] = {"threshold_on_raw_score": t, "source": "Stage 7 val_b_threshold (D29)"}
            entry["forced_binary"] = {"overall": _binary(binary_metrics(y, s, t)),
                                      **_subgroup_rows(y, subgroups, lambda m: _binary(binary_metrics(y[m], s[m], t)))}
            entry["ci"] = _model_cis(y, s, s, t, groups)
        out[model_id] = entry
    return out


def _h1(y, base, cnn, groups) -> dict[str, Any]:
    ci = _ci(lambda i: eer(y[i], base[i])[0] - eer(y[i], cnn[i])[0], len(y), groups,
             {"n_spoof": int(y.sum()), "n_bonafide": int((y == 0).sum())})
    supported = ci["estimate"] > 0 and ci["low"] > 0
    return {"rule": H1_RULE, "estimate": ci["estimate"], "ci": ci,
            "eer_baseline": eer(y, base)[0], "eer_bundle_cnn": eer(y, cnn)[0],
            "conditions": {"d_above_0": ci["estimate"] > 0, "ci_low_above_0": ci["low"] > 0},
            "outcome": "supported" if supported else "not supported"}


def _h5(y, subgroups, p, threshold, groups) -> dict[str, Any]:
    mp3_rows, other_rows = subgroups == "FAKE_mp3", subgroups == "FAKE_other"

    def recall(i, rows):
        i = i[rows[i]]  # the resampled rows of this subgroup; empty -> ValueError -> invalid resample
        return binary_metrics(y[i], p[i], threshold).recall

    mp3 = binary_metrics(y[mp3_rows], p[mp3_rows], threshold)
    other = binary_metrics(y[other_rows], p[other_rows], threshold)
    counts = {"FAKE_mp3": {"tp": mp3.tp, "fn": mp3.fn}, "FAKE_other": {"tp": other.tp, "fn": other.fn}}
    ci = _ci(lambda i: recall(i, mp3_rows) - recall(i, other_rows), len(y), groups, counts)
    exact = Fraction(mp3.tp, mp3.tp + mp3.fn) - Fraction(other.tp, other.tp + other.fn)
    conditions = {"d_at_least_0.05": exact >= H5_MIN_DIFFERENCE, "ci_low_above_0": ci["low"] > 0}
    return {"rule": H5_RULE, "estimate": ci["estimate"], "estimate_exact": f"{exact.numerator}/{exact.denominator}",
            "recall_FAKE_mp3": mp3.recall, "recall_FAKE_other": other.recall, "ci": ci,
            "conditions": conditions, "outcome": "supported" if all(conditions.values()) else "not supported"}


def _seeds(y, scores) -> dict[str, Any]:
    out = {}
    for name, metric in (("eer", lambda s: eer(y, s)[0]), ("roc_auc", lambda s: roc_auc(y, s))):
        summary = seed_summary({seed: metric(scores[CNN_IDS[seed]].raw_score) for seed in SEEDS})
        out[name] = {"mean": summary.mean, "sd": summary.sd, "n_seeds": summary.n_seeds,
                     "values": {str(seed): value for seed, value in summary.values}}
    out["note"] = "score level on raw_score; sample SD (ddof 1) of 3 seeds, a rough spread (Protocol §5)"
    return out


# --- Build and write ------------------------------------------------------------------------------

def build_report(*, run_dir, registry_path, manifest_path, risk_policy_path, stage7_dir, stage10_dir) -> dict:
    """Check the stored Stage 11 outputs, then compute the metrics; return the report dict.

    Raises ValueError (CheckFailed names the check) before any metric if a check fails. Writes nothing.
    """
    commit = code_version()["git_commit"]  # refuses a dirty worktree
    if not isinstance(commit, str) or not _HEX40.fullmatch(commit):
        raise ValueError(f"report_code_commit must be a full 40-hex SHA, got {commit!r}")
    run_dir, registry_path = Path(run_dir), Path(registry_path)
    checks: dict = {}

    report, report_bytes = _read_json(run_dir / RUN_REPORT)
    claim, claim_bytes = _read_json(run_dir / CLAIM_JSON)
    s7, s7_bytes = _read_json(Path(stage7_dir) / STAGE_REPORT)
    s10, s10_bytes = _read_json(Path(stage10_dir) / STAGE_REPORT)
    policy_bytes = Path(risk_policy_path).read_bytes()
    registry_bytes = registry_path.read_bytes()

    bundle_seed = _check_run_report(checks, report)
    bundle_id = CNN_IDS[bundle_seed]
    _check_claim(checks, report, claim, claim_bytes)
    _check_registry(checks, report, claim, registry_path)
    manifest, manifest_sha = _load_manifest(checks, Path(manifest_path), s10.get("f2_manifest_sha256"))
    clip_dir = run_dir / CLIP_SCORES_DIR
    scores = _read_scores(checks, report, clip_dir, manifest_sha)
    _check_same_clips(checks, scores)
    subgroups = _f2_join(checks, report, manifest, scores[BASELINE_ID])
    policy, baseline_threshold = _check_thresholds(checks, report, policy_bytes, s7, s10, bundle_seed)
    _rederive(checks, policy, scores[bundle_id], scores[BASELINE_ID], baseline_threshold)

    first = scores[BASELINE_ID]
    y = np.array([LABEL_CODES[x] for x in first.label], dtype=int)
    groups = list(first.group_id)
    bundle = scores[bundle_id]
    return {
        "schema_version": SCHEMA_VERSION,
        "report_code_commit": commit,
        "run": {"run_id": report["run_id"], "code_commit": report["code_commit"], "dataset": report["dataset"],
                "split": report["split"], "evaluation_purpose": report["evaluation_purpose"],
                "bundle_seed": bundle_seed, "bundle_model_id": bundle_id, "n_clips": len(y),
                "n_groups": len(set(groups)), "counts": report["counts"]},
        "inputs": {
            "run_report_sha256": _sha256(report_bytes),
            "claim_sha256": _sha256(claim_bytes),
            "registry": {"sha256": _sha256(registry_bytes), "lines": registry_bytes.count(b"\n")},
            "manifest_sha256": manifest_sha,
            "risk_policy_sha256": _sha256(policy_bytes),
            "stage7_run_report_sha256": _sha256(s7_bytes),
            "stage10_run_report_sha256": _sha256(s10_bytes),
            "clip_scores": {m: {k: report["clip_scores"]["files"][m][k] for k in ("csv_sha256", "sidecar_sha256")}
                            for m in MODEL_IDS},
        },
        "conventions": {"decision": "D57", **CONVENTIONS},
        "checks": checks,
        "models": _models(y, subgroups, scores, policy, baseline_threshold, groups, bundle_id),
        "seeds": _seeds(y, scores),
        "hypotheses": {
            "H1": _h1(y, first.raw_score, bundle.raw_score, groups),
            "H5": _h5(y, subgroups, bundle.p, policy.threshold, groups),
        },
    }


def _fmt(x, digits: int = 4) -> str:
    if x is None:
        return "undefined"
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, int):
        return str(x)
    return f"{x:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(r) + " |" for r in rows), ""]


def _ci_text(ci: dict) -> str:
    text = f"[{_fmt(ci['low'])}, {_fmt(ci['high'])}]"
    return text + " collapsed" if ci["collapsed"] else text


def _counts_text(counts: dict) -> str:
    return ", ".join(f"{k} {v}" if not isinstance(v, dict) else f"{k} " + "/".join(f"{a} {b}" for a, b in v.items())
                     for k, v in counts.items())


SCORE_PARTS = ("overall", *(f"REAL_vs_{f}" for f in FAKE_SUBGROUPS))
ROW_PARTS = ("overall", *SUBGROUPS)
CI_METRICS = ("eer", "roc_auc", "fpr", "fnr")


def _ordered(mapping: dict, order) -> list[tuple[str, Any]]:
    """Items in a fixed reading order (the JSON is written with sorted keys)."""
    return [(key, mapping[key]) for key in order if key in mapping]


def render_markdown(report: dict) -> str:
    """The Markdown report, rendered from the report JSON only (no number computed here)."""
    run = report["run"]
    models = _ordered(report["models"], MODEL_IDS)
    hypotheses = _ordered(report["hypotheses"], ("H1", "H5"))
    lines = [
        "# Stage 11 FoR test metrics report", "",
        "Generated by `src/evaluation/stage11_report.py` from the stored Stage 11 files only "
        "(no second test run, D31). Every number below comes from `metrics_report.json`; rates are "
        "shown to 4 decimals (full precision in the JSON); `undefined` = denominator 0.", "",
        "**Reported whatever the outcome (pre-registered).**", "",
        f"- Run `{run['run_id']}`, run commit `{run['code_commit']}`, report commit `{report['report_code_commit']}`",
        f"- Dataset `{run['dataset']}`, split `{run['split']}`, purpose `{run['evaluation_purpose']}`; "
        f"bundle seed {run['bundle_seed']} (`{run['bundle_model_id']}`)",
        f"- {run['n_clips']} clips in {run['n_groups']} D14 groups; labels "
        + ", ".join(f"{k} {v}" for k, v in run["counts"]["label"].items()) + "; subgroups "
        + ", ".join(f"{k} {v}" for k, v in run["counts"]["subgroup"].items()),
        f"- Integrity checks: {sum(c['passed'] for c in report['checks'].values())} of "
        f"{len(report['checks'])} passed; registry SHA-256 `{report['inputs']['registry']['sha256']}`, "
        f"{report['inputs']['registry']['lines']} lines",
        f"- Conventions: {report['conventions']['decision']} (listed in the JSON)", "",
        "## Hypotheses", "",
    ]
    rows = []
    for name, h in hypotheses:
        ci = h["ci"]
        rows.append([name, _fmt(h["estimate"]), _ci_text(ci), str(ci["n_invalid"]), _counts_text(ci["counts"]),
                     h["outcome"]])
    lines += _table(["Hypothesis", "d", "95% CI", "n_invalid", "Counts", "Outcome"], rows)
    for name, h in hypotheses:
        lines.append(f"- {name}: {h['rule']}.")
    h1, h5 = report["hypotheses"]["H1"], report["hypotheses"]["H5"]
    lines += [f"- H1: EER baseline-v1 {_fmt(h1['eer_baseline'])}, EER bundle CNN {_fmt(h1['eer_bundle_cnn'])}.",
              f"- H5: forced-binary recall FAKE_mp3 {_fmt(h5['recall_FAKE_mp3'])}, FAKE_other "
              f"{_fmt(h5['recall_FAKE_other'])}; d exactly {h5['estimate_exact']}.", ""]

    lines += ["## Score level (EER and ROC-AUC on raw_score, every model)", "",
              "The threshold at the EER point is descriptive only; the EER is not the error rate at any frozen "
              "threshold (harness Step 1).", ""]
    rows = []
    for model_id, m in models:
        for part, s in _ordered(m["score_level"], SCORE_PARTS):
            rows.append([model_id, part, str(s["n_bonafide"]), str(s["n_spoof"]), _fmt(s["eer"]),
                         _fmt(s["threshold_at_eer_point_descriptive"]), _fmt(s["roc_auc"])])
    lines += _table(["Model", "Rows", "n bonafide", "n spoof", "EER", "Threshold at EER point (descriptive)",
                     "ROC-AUC"], rows)

    lines += ["## Bootstrap 95% CIs (bundle CNN and baseline-v1)", "",
              "Group-clustered (D14), 2,000 resamples, seed 2026. A collapsed interval (low = high) does not "
              "mean there is no uncertainty (harness Step 4).", ""]
    rows = []
    for model_id, m in models:
        for metric, ci in _ordered(m.get("ci", {}), CI_METRICS):
            rows.append([model_id, metric, _fmt(ci["estimate"]), _ci_text(ci), str(ci["n_invalid"]),
                         str(ci["n_groups"]), _counts_text(ci["counts"])])
    lines += _table(["Model", "Metric", "Estimate", "95% CI", "n_invalid", "Groups", "Counts"], rows)

    lines += ["## Forced binary at the frozen threshold", ""]
    for model_id, m in models:
        if "forced_binary" not in m:
            continue
        frozen = m["frozen"]
        where = (f"p >= {frozen['threshold_on_p']!r} (policy {frozen['policy_version']})"
                 if "threshold_on_p" in frozen else f"raw_score >= {frozen['threshold_on_raw_score']!r} (D29)")
        lines += [f"**{model_id}**: synthetic iff {where}.", ""]
        rows = [[part, *(str(b[k]) for k in ("tp", "fp", "tn", "fn", "n_spoof", "n_bonafide")),
                 *(_fmt(b[k]) for k in ("accuracy", "precision", "recall", "f1", "fpr", "fnr"))]
                for part, b in _ordered(m["forced_binary"], ROW_PARTS)]
        lines += _table(["Rows", "TP", "FP", "TN", "FN", "n spoof", "n bonafide", "Accuracy", "Precision",
                         "Recall", "F1", "FPR", "FNR"], rows)

    for model_id, m in models:
        if "three_way" not in m:
            continue
        lines += ["## Three-way decisions (bundle CNN, stored predictions)", "",
                  f"Band on p: [{m['frozen']['threshold_on_p']!r} - {m['frozen']['half_width']!r}, "
                  f"{m['frozen']['threshold_on_p']!r} + {m['frozen']['half_width']!r}) is uncertain.", ""]
        rows = []
        for part, t in _ordered(m["three_way"], ROW_PARTS):
            b, s = t["confusion"]["bonafide"], t["confusion"]["spoof"]
            rows.append([part, *(str(x) for x in (*b, *s, t["n"], t["n_deferred"])),
                         *(_fmt(t[k]) for k in ("coverage", "deferral_rate", "deferral_rate_bonafide",
                                                 "deferral_rate_spoof"))])
        lines += _table(["Rows", "Bonafide auth.", "Bonafide synth.", "Bonafide uncert.", "Spoof auth.",
                         "Spoof synth.", "Spoof uncert.", "n", "Deferred", "Coverage", "Deferral",
                         "Deferral bonafide", "Deferral spoof"], rows)
        lines += ["Non-deferred cases only (synthetic = predicted spoof):", ""]
        rows = [[part, *(str(t["non_deferred"][k]) for k in ("tp", "fp", "tn", "fn")),
                 *(_fmt(t["non_deferred"][k]) for k in ("accuracy", "precision", "recall", "f1", "fpr", "fnr"))]
                for part, t in _ordered(m["three_way"], ROW_PARTS)]
        lines += _table(["Rows", "TP", "FP", "TN", "FN", "Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"],
                        rows)

    seeds = report["seeds"]
    lines += ["## CNN seeds (score level, separate row; Protocol §5)", "", seeds["note"] + ".", ""]
    rows = [[f"seed {seed}", _fmt(seeds["eer"]["values"][seed]), _fmt(seeds["roc_auc"]["values"][seed])]
            for seed in seeds["eer"]["values"]]
    rows.append(["mean ± SD", f"{_fmt(seeds['eer']['mean'])} ± {_fmt(seeds['eer']['sd'])}",
                 f"{_fmt(seeds['roc_auc']['mean'])} ± {_fmt(seeds['roc_auc']['sd'])}"])
    lines += _table(["Seed", "EER", "ROC-AUC"], rows)
    return "\n".join(lines)


def _create(path: Path, data: bytes) -> None:
    with open(path, "xb") as handle:  # exclusive creation, binary: same bytes on every OS
        handle.write(data)


def write_report(report: dict, out_dir) -> tuple[Path, Path]:
    """Write metrics_report.json and metrics_report.md (rendered from the JSON) into a new folder."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError(f"output folder already exists: {out_dir}")
    text = _json_text(report)  # strict JSON: NaN would raise here
    markdown = render_markdown(json.loads(text))
    out_dir.mkdir(parents=True)
    json_path, md_path = out_dir / REPORT_JSON, out_dir / REPORT_MD
    _create(json_path, text.encode("utf-8"))
    _create(md_path, markdown.encode("utf-8"))
    return json_path, md_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, type=Path, help="The folder the Stage 11 score step wrote")
    parser.add_argument("--registry", required=True, type=Path, help="Run registry JSON Lines file")
    parser.add_argument("--manifest", required=True, type=Path, help="Approved F2 FoR manifest")
    parser.add_argument("--risk-policy", required=True, type=Path, help="risk_policy.json (Stage 10)")
    parser.add_argument("--stage7-evidence", required=True, type=Path, help="docs/audit_evidence/stage7_baseline")
    parser.add_argument("--stage10-evidence", required=True, type=Path, help="docs/audit_evidence/stage10_risk")
    parser.add_argument("--out", required=True, type=Path, help="New output folder")
    args = parser.parse_args(argv)
    if args.out.exists():  # before any work
        raise ValueError(f"output folder already exists: {args.out}")
    report = build_report(run_dir=args.run_dir, registry_path=args.registry, manifest_path=args.manifest,
                          risk_policy_path=args.risk_policy, stage7_dir=args.stage7_evidence,
                          stage10_dir=args.stage10_evidence)
    json_path, md_path = write_report(report, args.out)
    print(f"{json_path}\n{md_path}")


if __name__ == "__main__":
    main()
