"""Official Stage 10 risk-policy run (Protocol v1.0 §6; D2, D9, D12, D44, D45): val-A check, val-B fit.

Loads the bundle-seed CNN checkpoint (D45: that seed only) against the checkpoint_sha256 recorded in
the committed Stage 8 evidence, re-scores val-A on CPU to reproduce the checkpoint's val-A EER
exactly (verification only, no decision), then scores val-B and fits the risk policy on it with
fit_risk_policy. Writes into a new folder:
- risk_policy.json: RiskPolicy.to_json().
- val_b_scores.npz: paths, labels, raw score (float64) and calibrated p per val-B clip (non-pickle).
- diagnostics_report.json: descriptive only (pre-declared 2026-10-28); nothing reads it back.
- run_report.json: provenance binding the policy to its inputs (code commit, F3 profile, F2
  manifest, cache hashes, checkpoint hash) and the headline numbers.

Only val-A and val-B exist here: there is no split argument and no train or test path; the FoR
test split is evaluated once at Stage 11 (D31). Nothing is logged to W&B (D40); the reports are the
record. Nothing is written unless every step succeeds.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.evaluation.metrics import eer, roc_auc
from src.features.logmel_cache import load_logmel_cache
from src.models.baseline import _write_npz
from src.models.cnn_checkpoint import load_checkpoint
from src.preprocessing.profile import load_profile
from src.preprocessing.validation import (
    F2_MANIFEST_SHA256,
    code_version,
    load_f2_manifest,
    sha256_file,
)
from src.risk.decisions import AUTHENTIC, DECISIONS, SYNTHETIC, three_way_decisions
from src.risk.policy import RiskPolicy
from src.training.cnn_config import load_cnn_config
from src.training.normalization import apply_bin_stats
from src.training.train_cnn import _score
from src.tuning.baseline_run import FAKE_SUBGROUPS, _json_text, _subgroups
from src.tuning.risk_fit import fit_risk_policy

SCHEMA_VERSION = 1
CHECK_SPLIT = "val-A"  # reproduction check only (verification, no decision)
FIT_SPLIT = "val-B"    # the risk policy is fitted here only (D2, D9)
STAGE8_RUN_REPORT = "run_report.json"
STAGE8_SELECTION_REPORT = "selection_report.json"
RISK_POLICY = "risk_policy.json"
VAL_B_SCORES = "val_b_scores.npz"
DIAGNOSTICS_REPORT = "diagnostics_report.json"
RUN_REPORT = "run_report.json"
DEVICE = "cpu"
DIAGNOSTICS_PURPOSE = (
    "descriptive only; not used for the calibration, threshold, band or any decision "
    "(pre-declared 2026-10-28)"
)
CLASSES = {"bonafide": 0, "spoof": 1}


def environment() -> dict[str, str]:
    """Python and package versions for run_report.json (scores depend on NumPy and torch)."""
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}


def bundle_checkpoint(stage8_evidence_dir) -> tuple[int, str]:
    """(bundle seed, its checkpoint_sha256) from the committed Stage 8 evidence; raise if inconsistent."""
    folder = Path(stage8_evidence_dir)
    run_report = json.loads((folder / STAGE8_RUN_REPORT).read_text(encoding="utf-8"))
    selection = json.loads((folder / STAGE8_SELECTION_REPORT).read_text(encoding="utf-8"))
    seed = run_report["bundle_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"Stage 8 bundle_seed must be an int, got {seed!r}")
    if selection["bundle_seed"] != seed:
        raise ValueError(f"Stage 8 evidence disagrees on the bundle seed: run_report {seed}, "
                         f"selection_report {selection['bundle_seed']!r}")
    hashes = run_report["checkpoint_sha256"]
    if str(seed) not in hashes:
        raise ValueError(f"Stage 8 run_report has no checkpoint_sha256 for bundle seed {seed}")
    return seed, hashes[str(seed)]


def _split_rows(manifest, split: str, paths: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Subgroups of the cached paths; raise unless they are exactly the F2 rows of split, same labels."""
    subgroups = _subgroups(manifest, split, paths, labels)  # every path in F2, this split, same label
    expected = manifest.loc[manifest["split"] == split, "path"]
    absent = sorted(set(expected) - set(paths.tolist()))
    if absent:
        raise ValueError(f"{split}: {len(absent)} F2 rows are missing from the cache, first {absent[0]!r}")
    return subgroups


def _load_split(cache_dir, split: str, profile, profile_config_sha256: str, manifest):
    """Load one verified log-Mel cache bound to the profile file and complete against F2."""
    features, labels, paths, meta = load_logmel_cache(
        Path(cache_dir) / split, split, profile, expected_f2_manifest_sha256=F2_MANIFEST_SHA256
    )
    if meta["profile_config_sha256"] != profile_config_sha256:
        raise ValueError(
            f"{split}: cache was built from profile config {meta['profile_config_sha256']}, "
            f"profile file is {profile_config_sha256}"
        )
    subgroups = _split_rows(manifest, split, paths, labels)
    return features, labels, paths, subgroups, meta


def score_clips(model, mean, std, features, batch_size: int) -> np.ndarray:
    """Raw scores (float64 logits) on CPU by the training path: apply_bin_stats, from_numpy, _score."""
    x = torch.from_numpy(apply_bin_stats(features, mean, std))
    return _score(model, x, batch_size, torch.device(DEVICE))


def _per_class(values: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    return {name: int(values[labels == code].sum()) for name, code in CLASSES.items()}


def diagnostics(labels, scores, policy: RiskPolicy, subgroups) -> dict[str, Any]:
    """Descriptive val-B diagnostics at the fitted policy (pre-declared 2026-10-28); a pure function.

    Nothing here feeds back into the fit. EER and ROC-AUC are reported on raw scores and on p, not
    asserted equal: p can tie where the sigmoid saturates (p == 0 or 1 in float64).
    """
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(subgroups)
    p = policy.calibration.transform(s)
    bona, spoof = y == 0, y == 1
    decisions = three_way_decisions(p, policy.threshold, policy.half_width)
    forced = three_way_decisions(p, policy.threshold, 0.0)

    out: dict[str, Any] = {
        "purpose": DIAGNOSTICS_PURPOSE,
        "counts": {"n_bonafide": int(bona.sum()), "n_spoof": int(spoof.sum())},
        "score_level": {
            "raw": {"eer": eer(y, s)[0], "roc_auc": roc_auc(y, s)},
            "p": {"eer": eer(y, p)[0], "roc_auc": roc_auc(y, p)},
            "n_p_equal_0": _per_class(p == 0.0, y),
            "n_p_equal_1": _per_class(p == 1.0, y),
        },
        "separation": {
            "max_bonafide_raw": float(s[bona].max()),
            "min_spoof_raw": float(s[spoof].min()),
            "max_bonafide_p": float(p[bona].max()),
            "min_spoof_p": float(p[spoof].min()),
            "separable": bool(s[bona].max() < s[spoof].min()),
        },
        "policy": {
            "a": policy.calibration.a, "b": policy.calibration.b, "threshold": policy.threshold,
            "half_width": policy.half_width, "target_deferral": policy.target_deferral,
            "deferral_on_fit": policy.deferral_on_fit,
        },
        "decisions": {name: {d: int((decisions[y == code] == d).sum()) for d in DECISIONS}
                      for name, code in CLASSES.items()},
        "forced_binary": {
            "false_positives": int((forced[bona] == SYNTHETIC).sum()),
            "false_negatives": int((forced[spoof] == AUTHENTIC).sum()),
        },
        "subgroups": {
            "counts": {name: int((groups == name).sum()) for name in ("REAL", *FAKE_SUBGROUPS)},
        },
    }
    for fake in FAKE_SUBGROUPS:
        mask = (groups == "REAL") | (groups == fake)
        out["subgroups"][f"REAL_vs_{fake}"] = {
            "eer": eer(y[mask], s[mask])[0],
            "n_bonafide": int((groups == "REAL").sum()),
            "n_spoof": int((groups == fake).sum()),
        }
    return out


def run(cache_dir, checkpoint_dir, profile_path, config_path, manifest_path, stage8_evidence_dir,
        out_dir, *, version: str) -> dict[str, Any]:
    """Verify the bundle checkpoint on val-A, fit the risk policy on val-B, write outputs; return run_report.

    cache_dir holds val-A/ and val-B/ (one load_logmel_cache folder each). out_dir must not exist;
    on any failure it is removed.
    """
    commit = code_version()["git_commit"]  # before reading inputs or creating outputs
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError(f"output folder already exists: {out_dir}")

    bundle_seed, expected_sha256 = bundle_checkpoint(stage8_evidence_dir)
    model, mean, std, ckpt = load_checkpoint(checkpoint_dir, expected_sha256=expected_sha256)
    if ckpt["seed"] != bundle_seed:  # D45: the bundle seed only
        raise ValueError(f"checkpoint seed {ckpt['seed']} is not the Stage 8 bundle seed {bundle_seed}")

    profile_path, config_path, manifest_path = Path(profile_path), Path(config_path), Path(manifest_path)
    profile = load_profile(profile_path)
    batch_size = load_cnn_config(config_path).batch_size
    manifest = load_f2_manifest(manifest_path)
    profile_config_sha256 = sha256_file(profile_path)

    # Reproduction check: weights, normalization and scoring path must give the recorded val-A EER.
    val_a = _load_split(cache_dir, CHECK_SPLIT, profile, profile_config_sha256, manifest)
    val_a_eer = eer(val_a[1], score_clips(model, mean, std, val_a[0], batch_size))[0]
    if val_a_eer != ckpt["val_a_eer"]:
        raise ValueError(f"val-A reproduction failed: recomputed EER {val_a_eer!r}, "
                         f"checkpoint records {ckpt['val_a_eer']!r}")

    features, labels, paths, subgroups, val_b_meta = _load_split(
        cache_dir, FIT_SPLIT, profile, profile_config_sha256, manifest
    )
    scores = score_clips(model, mean, std, features, batch_size)
    if not np.array_equal(scores, score_clips(model, mean, std, features, batch_size)):
        raise ValueError("val-B scores differ between two CPU scoring passes")

    policy = fit_risk_policy(labels, scores, version=version)
    policy_text = policy.to_json()
    if fit_risk_policy(labels, scores, version=version).to_json() != policy_text:
        raise ValueError("risk policy refit from the same scores is not byte-identical")
    diagnostics_report = diagnostics(labels, scores, policy, subgroups)
    p = policy.calibration.transform(scores)

    out_dir.mkdir(parents=True)
    try:
        policy_path = out_dir / RISK_POLICY
        policy_path.write_text(policy_text, encoding="utf-8")
        scores_path = out_dir / VAL_B_SCORES
        _write_npz(scores_path, {"paths": paths, "labels": labels, "score": scores, "p": p})
        run_report = {
            "schema_version": SCHEMA_VERSION,
            "version": version,
            "code_commit": commit,
            "environment": environment(),
            "profile_fingerprint": profile.fingerprint(),
            "profile_config_sha256": profile_config_sha256,
            "f2_manifest_sha256": sha256_file(manifest_path),
            "caches": {
                split: {"npz_sha256": meta["npz_sha256"], "n_rows": meta["n_rows"]}
                for split, meta in ((CHECK_SPLIT, val_a[4]), (FIT_SPLIT, val_b_meta))
            },
            "bundle_seed": bundle_seed,
            "checkpoint_sha256": expected_sha256,
            "n_blocks": ckpt["n_blocks"],
            "scoring": {"device": DEVICE, "batch_size": batch_size},
            "val_a_eer_recomputed": val_a_eer,
            "risk_policy_sha256": sha256_file(policy_path),
            "val_b_scores_sha256": sha256_file(scores_path),
            "headline": {
                "a": policy.calibration.a,
                "b": policy.calibration.b,
                "threshold": policy.threshold,
                "half_width": policy.half_width,
                "deferral_on_fit": policy.deferral_on_fit,
                "n_bonafide": policy.n_bonafide_fit,
                "n_spoof": policy.n_spoof_fit,
            },
        }
        (out_dir / DIAGNOSTICS_REPORT).write_text(_json_text(diagnostics_report), encoding="utf-8")
        (out_dir / RUN_REPORT).write_text(_json_text(run_report), encoding="utf-8")
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return run_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", required=True, type=Path,
                        help="Folder holding the val-A and val-B log-Mel caches")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Bundle-seed checkpoint folder")
    parser.add_argument("--profile", required=True, type=Path, help="F3 preprocessing profile YAML")
    parser.add_argument("--config", required=True, type=Path, help="configs/cnn.yaml (batch_size)")
    parser.add_argument("--manifest", required=True, type=Path, help="Approved F2 FoR manifest")
    parser.add_argument("--stage8-evidence", required=True, type=Path,
                        help="docs/audit_evidence/stage8_cnn (bundle seed and checkpoint_sha256)")
    parser.add_argument("--out", required=True, type=Path, help="New output folder")
    parser.add_argument("--version", required=True, help="Risk policy version, e.g. risk-v1")
    args = parser.parse_args(argv)
    report = run(args.cache_dir, args.checkpoint, args.profile, args.config, args.manifest,
                 args.stage8_evidence, args.out, version=args.version)
    print(_json_text(report), end="")


if __name__ == "__main__":
    main()
