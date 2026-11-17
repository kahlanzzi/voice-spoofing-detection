"""Stage 11 metrics report for the model v2 set, from the stored files only (Protocol v1.0 §7 addendum; D48 (7)).

The v2 counterpart of src/evaluation/stage11_report.py, which stays unchanged (its output for the
real v1 run is committed and reproduced, Entry 161, and its checks are bound to the v1 model IDs).
Reads the output folder of src.evaluation.stage11_v2_run score (run_report.json, claim.json and the
three per-clip score files), the run registry, the F2 manifest, and the committed v2 search and
risk-v2 evidence (the risk policy is <risk-evidence>/risk_policy.json, the file the runner used). It
reads no audio, no feature cache, no checkpoint and no model, and never calls the runner (D31).

Scope (Donia, option A, 2026-11-17, before any v2 test result): D57 applied to the v2 model set as
written, with no hypothesis. H1 and H5 are v1 hypotheses (Protocol §1); v2 is compared with v1 in H8
(Stage 13) and H9 (Stage 14). No baseline row and no v1-v2 comparison here.

Order:
1. Integrity checks, each recorded as expected / actual / passed and raising CheckFailed naming the
   check: run report (architecture, v2 model set and roles), claim, registry (one started and one
   completed record for this run_id with the v2 model set), each score file against its sidecar and
   the run report (and no other file in the folder), the same clips in all three files, the F2
   join, the evidence (search winner and bundle seed, per-seed checkpoint hashes, risk-v2 binding
   and policy hash, F2 hash), and an independent re-derivation of the stored p and prediction.
2. Metrics under D57 mapped to v2 (CONVENTIONS_V2): score level for the three seeds; forced-binary
   and three-way metrics and cluster-bootstrap CIs for the bundle-seed model; the seed rows.
3. write_report: metrics_report.json (strict JSON) and metrics_report.md, rendered from the JSON
   only, into a new folder.

The model-independent helpers are stage11_report's own, imported, not copied; only what is bound to
the v1 model IDs, roles or evidence is written here. The split is read from run_report.json and must
equal claim.json and the registry record; this module never names it. Torch-free: the v2 runner's
constants are re-declared here, and a test checks they equal the runner's.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.clip_scores import read_clip_scores
from src.evaluation.metrics import binary_metrics, eer, roc_auc, three_way_metrics
from src.evaluation.run_registry import COMPLETED, STARTED, RunRegistry
from src.evaluation.seeds import seed_summary
from src.evaluation.stage11_report import (
    _HEX32,
    _HEX40,
    CI_METRICS,
    REPORT_JSON,
    REPORT_MD,
    ROW_PARTS,
    SCORE_PARTS,
    CheckFailed,
    _binary,
    _check,
    _check_claim,
    _ci_text,
    _counts_text,
    _create,
    _digest,
    _f2_join,
    _fail,
    _fmt,
    _model_cis,
    _ordered,
    _read_json,
    _record,
    _score_level,
    _sha256,
    _subgroup_rows,
    _table,
    _three_way,
)
from src.features.mfcc_cache import LABEL_CODES
from src.preprocessing.validation import F2_COLUMNS, F2_MANIFEST_SHA256, _safe_relative, code_version
from src.risk.decisions import AUTHENTIC, SYNTHETIC, UNCERTAIN
from src.risk.policy import RiskPolicy
from src.tuning.baseline_run import FAKE_SUBGROUPS, _json_text

__all__ = ["CheckFailed", "build_report", "write_report", "render_markdown", "search_bundle", "main"]

# Re-declared from src.evaluation.stage11_v2_run, src.models.v2_checkpoint, src.tuning.v2_search and
# src.tuning.risk_run (all import torch); a test checks equality.
RUN_SCHEMA_VERSION = 1
DATASET = "for"
PURPOSE = "stage11_test"
SEEDS = (0, 1, 2)
ARCHITECTURES = ("cnn-aug", "crnn", "aasist")  # D49 candidate order
ROLE_BUNDLE = "v2_bundle_seed"
ROLE_OTHER = "v2_other_seed"
CLIP_SCORES_DIR = "clip_scores"
RUN_REPORT = "run_report.json"
CLAIM_JSON = "claim.json"
SEARCH_RUN_REPORT = "run_report.json"
SEARCH_SELECTION_REPORT = "selection_report.json"
RISK_RUN_REPORT = "run_report.json"
RISK_POLICY = "risk_policy.json"


def model_ids_v2(winner: str) -> tuple[str, ...]:
    """The v2 model set (D48 (7)): `<winner>-v2-seed<k>` for k = 0, 1, 2, in seed order."""
    if winner not in ARCHITECTURES:
        raise ValueError(f"winner must be one of {ARCHITECTURES}, got {winner!r}")
    return tuple(f"{winner}-v2-seed{k}" for k in SEEDS)


def model_roles(winner: str, bundle_seed: int) -> dict[str, str]:
    """{model_id: role}: v2_bundle_seed for the bundle seed (D45), v2_other_seed for the others."""
    if bundle_seed not in SEEDS:
        raise ValueError(f"bundle_seed must be one of {SEEDS}, got {bundle_seed!r}")
    return {m: ROLE_BUNDLE if k == bundle_seed else ROLE_OTHER for k, m in zip(SEEDS, model_ids_v2(winner))}


def run_dir_name(candidate: str, seed: int) -> str:
    """The v2 search run folder name, the key of the search run report's checkpoint_sha256."""
    return f"{candidate}_seed{seed}"


SCHEMA_VERSION = 1
SUBGROUPS = ("REAL", *FAKE_SUBGROUPS)
APPROVED_F2_MANIFEST_SHA256 = F2_MANIFEST_SHA256

# D57 mapped to the v2 model set (Protocol §7 addendum: "the same storage and reporting rules apply").
CONVENTIONS_V2 = {
    "1_model": "the model = the bundle-seed v2 model (role v2_bundle_seed, seed from run_report.json); the "
               "other two seeds at score level only (Protocol §5, D45)",
    "2_score_level": "EER and ROC-AUC on raw_score for all three seeds, never on p",
    "3_subgroups": "as D57 (3): score level: REAL vs FAKE_mp3 and REAL vs FAKE_other; decision level: each "
                   "subgroup alone (binary_metrics / three_way_metrics on the subset)",
    "4_decision_level": "bundle-seed v2 model: forced binary = binary_metrics(labels, p, risk-v2 threshold), "
                        "three-way = three_way_metrics(labels, stored prediction); no baseline row (baseline-v1 "
                        "was scored in the v1 claim)",
    "5_h5": "not applicable: no hypothesis is tested at v2 Stage 11 (hypotheses_note)",
    "6_h1": "not applicable: no hypothesis is tested at v2 Stage 11 (hypotheses_note)",
    "7_ci": "as D57 (7) for the bundle-seed v2 model only: bootstrap_ci defaults (95% percentile, 2,000 "
            "resamples, seed 2026), clusters = group_id (D14), on all rows; overall EER, ROC-AUC, forced-binary "
            "FPR and FNR; counts, n_invalid and collapsed (low == high) reported; subgroup rows: point estimates "
            "with counts, no CIs",
    "8_seeds": "as D57 (8): seed_summary of test EER and ROC-AUC over v2 seeds 0, 1, 2 (mean, sample SD, "
               "values), a separate row next to the bundle-seed CIs",
    "9_undefined": "as D57 (9): undefined rates (denominator 0) are JSON null; strict JSON; the EER threshold "
                   "is descriptive only; realised FPR/FNR are at the frozen risk-v2 threshold",
}
CONVENTIONS_SCOPE = ("D57 applied to the v2 model set as written, with no hypothesis (Donia, option A, "
                     "2026-11-17, before any v2 test result; Protocol §7 addendum)")
HYPOTHESES_NOTE = ("No hypothesis is tested at v2 Stage 11: H1 and H5 are v1 hypotheses (Protocol §1); v2 is "
                   "compared with v1 in H8 (Stage 13) and H9 (Stage 14).")


# --- Evidence: the search winner and bundle seed (torch-free) -------------------------------------

def search_bundle(evidence_dir) -> tuple[str, int, str]:
    """(winner, bundle seed, its checkpoint_sha256) from the v2 search evidence; raise if inconsistent.

    The agreement rule of src.tuning.v2_risk_run.bundle_checkpoint (which imports torch), written
    out here; a test checks both give the same answer and refuse the same evidence.
    """
    folder = Path(evidence_dir)
    for name in (SEARCH_RUN_REPORT, SEARCH_SELECTION_REPORT):
        if not (folder / name).is_file():
            raise ValueError(f"v2 search evidence {folder / name} is missing; the risk run needs the "
                             f"committed search reports")
    run_report = json.loads((folder / SEARCH_RUN_REPORT).read_text(encoding="utf-8"))
    selection = json.loads((folder / SEARCH_SELECTION_REPORT).read_text(encoding="utf-8"))
    winner, seed = run_report["selected"], run_report["bundle_seed"]
    if winner not in ARCHITECTURES:
        raise ValueError(f"v2 search selected must be one of {ARCHITECTURES}, got {winner!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"v2 search bundle_seed must be an int, got {seed!r}")
    if selection["selected"] != winner:
        raise ValueError(f"v2 search evidence disagrees on the winner: run_report {winner!r}, "
                         f"selection_report {selection['selected']!r}")
    if selection["bundle_seed"] != seed:
        raise ValueError(f"v2 search evidence disagrees on the bundle seed: run_report {seed}, "
                         f"selection_report {selection['bundle_seed']!r}")
    name = run_dir_name(winner, seed)
    hashes = run_report["checkpoint_sha256"]
    if name not in hashes:
        raise ValueError(f"v2 search run_report has no checkpoint_sha256 for {name}")
    return winner, seed, hashes[name]


# --- 1. Integrity checks --------------------------------------------------------------------------

def _check_run_report(checks: dict, report: dict) -> tuple[str, int, tuple[str, ...]]:
    """Check 1: run_report.json; return (architecture, bundle seed, model IDs in seed order)."""
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
    architecture = report.get("architecture")
    _record(checks, "run_report.architecture", f"one of {list(ARCHITECTURES)}", architecture,
            isinstance(architecture, str) and architecture in ARCHITECTURES)
    model_ids = model_ids_v2(architecture)
    model_set = report.get("model_set")
    ids = sorted(model_set) if isinstance(model_set, dict) else None
    _check(checks, "run_report.model_set.ids", sorted(model_ids), ids)  # a v1 or baseline ID is refused here
    roles = sorted(str(model_set[m].get("role")) for m in model_ids)
    _check(checks, "run_report.model_set.roles", sorted([ROLE_BUNDLE, ROLE_OTHER, ROLE_OTHER]), roles)
    seed = report.get("bundle_seed")
    _record(checks, "run_report.bundle_seed", f"one of {list(SEEDS)}", seed,
            not isinstance(seed, bool) and seed in SEEDS)
    _check(checks, "run_report.bundle_seed_role", ROLE_BUNDLE, model_set[model_ids[SEEDS.index(seed)]]["role"])
    return architecture, seed, model_ids


def _check_claim_v2(checks: dict, report: dict, claim: dict, claim_bytes: bytes, model_ids) -> None:
    """Check 2: stage11_report's claim check (run, commit, split, dataset, purpose, SHA-256, results_ref),
    then the v2 fields the v2 claim adds."""
    _check_claim(checks, report, claim, claim_bytes)
    _check(checks, "claim.architecture", report["architecture"], claim.get("architecture"))
    _check(checks, "claim.bundle_seed", report["bundle_seed"], claim.get("bundle_seed"))
    _check(checks, "claim.model_ids", sorted(model_ids), claim.get("model_ids"))


def _check_registry(checks: dict, report: dict, claim: dict, registry_path: Path, model_ids) -> None:
    """Check 3: exactly one started and one completed record for this run_id, with the v2 model set.

    Other records (the v1 pair, later claims) are not this run's and are not judged here.
    """
    try:
        records = RunRegistry(registry_path).records()
    except ValueError as error:
        raise _fail(checks, "registry.load", error) from None
    mine = [r for r in records if r.run_id == report["run_id"]]
    _check(checks, "registry.records", sorted([COMPLETED, STARTED]), sorted(r.status for r in mine))
    expected_key = [report["dataset"], report["split"], report["evaluation_purpose"], sorted(model_ids)]
    for record in mine:
        _check(checks, f"registry.{record.status}.key", expected_key,
               [record.dataset, record.split, record.evaluation_purpose, list(record.model_set)])
        _check(checks, f"registry.{record.status}.code_commit", report["code_commit"], record.code_commit)
    completed = next(r for r in mine if r.status == COMPLETED)
    _check(checks, "registry.completed.results_ref", claim["results_ref"], completed.results_ref)


def _load_manifest(checks: dict, path: Path, expected_sha256) -> tuple[pd.DataFrame, str]:
    """The F2 manifest as load_f2_manifest reads it, bound to risk-v2's hash and the approved value.

    stage11_report._load_manifest names its check after Stage 10 and reads that module's approved
    value, so the v2 one is written here (same reading rule).
    """
    data = path.read_bytes()
    sha = _sha256(data)
    _check(checks, "manifest.sha256_risk_v2", expected_sha256, sha)
    _check(checks, "manifest.sha256_approved", APPROVED_F2_MANIFEST_SHA256, sha)
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    _check(checks, "manifest.columns", list(F2_COLUMNS), list(manifest.columns))
    manifest["path"] = manifest["path"].map(_safe_relative)
    _check(checks, "manifest.unique_paths", 0, int(manifest["path"].duplicated().sum()))
    return manifest, sha


def _read_scores(checks: dict, report: dict, clip_dir: Path, manifest_sha256: str, model_ids) -> dict:
    """Check 4: the three score files read and verified, bound to the run report and the F2 manifest;
    the folder holds exactly their CSV and sidecar (no baseline or v1 file)."""
    files = report.get("clip_scores", {})
    _check(checks, "clip_scores.folder", CLIP_SCORES_DIR, files.get("folder"))
    _check(checks, "clip_scores.models", sorted(model_ids), sorted(files.get("files", {})))
    present = sorted(p.name for p in clip_dir.iterdir()) if clip_dir.is_dir() else None
    _check(checks, "clip_scores.folder_files", sorted(f"{m}.{ext}" for m in model_ids for ext in ("csv", "json")),
           present)
    out = {}
    for model_id in model_ids:
        expected = files["files"][model_id]
        try:
            scores = read_clip_scores(clip_dir, model_id)
        except ValueError as error:
            raise _fail(checks, f"clip_scores.{model_id}.read", error) from None
        _check(checks, f"clip_scores.{model_id}.csv_sha256", expected.get("csv_sha256"),
               _sha256((clip_dir / f"{model_id}.csv").read_bytes()))
        _check(checks, f"clip_scores.{model_id}.sidecar_sha256", expected.get("sidecar_sha256"),
               _sha256((clip_dir / f"{model_id}.json").read_bytes()))
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


def _check_same_clips(checks: dict, scores: dict, model_ids) -> None:
    """Check 5: the same clip IDs, labels and groups, in the same order, in all three files."""
    first = scores[model_ids[0]]
    for name in ("clip_id", "label", "group_id"):
        expected = _digest(getattr(first, name))
        for model_id in model_ids[1:]:
            _check(checks, f"same_clips.{model_id}.{name}", expected, _digest(getattr(scores[model_id], name)))


def _check_evidence(checks: dict, report: dict, model_ids, search_dir: Path, search_bytes: dict,
                    risk: dict, policy_bytes: bytes) -> RiskPolicy:
    """Check 7: the run is the search winner's set and its bundle model is the one risk-v2 was fitted on."""
    architecture, bundle_seed = report["architecture"], report["bundle_seed"]
    model_set = report["model_set"]
    try:
        winner, seed, _ = search_bundle(search_dir)
    except (ValueError, KeyError, TypeError) as error:
        raise _fail(checks, "search.bundle_checkpoint", error) from None
    _check(checks, "search.architecture", winner, architecture)
    _check(checks, "search.bundle_seed", seed, bundle_seed)
    hashes = json.loads(search_bytes[SEARCH_RUN_REPORT].decode("utf-8")).get("checkpoint_sha256", {})
    for k, model_id in zip(SEEDS, model_ids):
        _check(checks, f"search.checkpoint_sha256.{model_id}", hashes.get(run_dir_name(architecture, k)),
               model_set[model_id]["artifact_sha256"])
    bundle_id = model_ids[SEEDS.index(bundle_seed)]
    _check(checks, "risk_v2.architecture", risk.get("architecture"), architecture)
    _check(checks, "risk_v2.bundle_seed", risk.get("bundle_seed"), bundle_seed)
    _check(checks, "risk_v2.checkpoint_sha256", risk.get("checkpoint_sha256"), model_set[bundle_id]["artifact_sha256"])
    _check(checks, "risk_v2.v2_search_evidence", risk.get("v2_search_evidence"),
           {name: _sha256(data) for name, data in search_bytes.items()})
    _check(checks, "risk_policy.sha256", risk.get("risk_policy_sha256"), _sha256(policy_bytes))
    return RiskPolicy.from_json(policy_bytes.decode("utf-8"))


def _rederive(checks: dict, policy: RiskPolicy, scores: dict, bundle_id: str, model_ids) -> None:
    """Check 8: the stored bundle p and prediction follow from the stored raw score (independent of the
    runner); the other seeds hold neither (D45)."""
    bundle = scores[bundle_id]
    p = policy.calibration.transform(bundle.raw_score)
    _check(checks, "rederive.bundle.p_bytes", _sha256(p.tobytes()), _sha256(bundle.p.tobytes()))
    # The band rule of src/risk/decisions.py three_way_decisions, written out:
    # synthetic if p >= threshold + half_width; authentic if p < threshold - half_width; else uncertain.
    upper, lower = policy.threshold + policy.half_width, policy.threshold - policy.half_width
    expected = np.where(bundle.p >= upper, SYNTHETIC, np.where(bundle.p < lower, AUTHENTIC, UNCERTAIN))
    _check(checks, "rederive.bundle.prediction_mismatches", 0,
           int(np.sum(expected != np.array(bundle.prediction, dtype=object))))
    for model_id in model_ids:
        if model_id != bundle_id:
            other = scores[model_id]
            _check(checks, f"rederive.{model_id}.p_and_prediction_absent", True,
                   other.p is None and other.prediction is None)


# --- 2. Metrics -----------------------------------------------------------------------------------

def _models(y, subgroups, scores, policy, groups, bundle_id, model_ids) -> dict[str, Any]:
    out = {}
    everything = np.ones(len(y), dtype=bool)
    for model_id in model_ids:
        clip = scores[model_id]
        s = clip.raw_score
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
        out[model_id] = entry
    return out


def _seeds(y, scores, model_ids) -> dict[str, Any]:
    out = {}
    for name, metric in (("eer", lambda s: eer(y, s)[0]), ("roc_auc", lambda s: roc_auc(y, s))):
        summary = seed_summary({seed: metric(scores[model_id].raw_score) for seed, model_id in zip(SEEDS, model_ids)})
        out[name] = {"mean": summary.mean, "sd": summary.sd, "n_seeds": summary.n_seeds,
                     "values": {str(seed): value for seed, value in summary.values}}
    out["note"] = "score level on raw_score; sample SD (ddof 1) of 3 seeds, a rough spread (Protocol §5)"
    return out


# --- Build and write ------------------------------------------------------------------------------

def build_report(*, run_dir, registry_path, manifest_path, search_evidence_dir, risk_evidence_dir) -> dict:
    """Check the stored v2 Stage 11 outputs, then compute the metrics; return the report dict.

    Raises ValueError (CheckFailed names the check) before any metric if a check fails. Writes nothing.
    """
    commit = code_version()["git_commit"]  # refuses a dirty worktree
    if not isinstance(commit, str) or not _HEX40.fullmatch(commit):
        raise ValueError(f"report_code_commit must be a full 40-hex SHA, got {commit!r}")
    run_dir, registry_path = Path(run_dir), Path(registry_path)
    search_dir, risk_dir = Path(search_evidence_dir), Path(risk_evidence_dir)
    checks: dict = {}

    report, report_bytes = _read_json(run_dir / RUN_REPORT)
    claim, claim_bytes = _read_json(run_dir / CLAIM_JSON)
    risk, risk_bytes = _read_json(risk_dir / RISK_RUN_REPORT)
    search_bytes = {name: (search_dir / name).read_bytes() for name in (SEARCH_RUN_REPORT, SEARCH_SELECTION_REPORT)}
    policy_bytes = (risk_dir / RISK_POLICY).read_bytes()  # one read: hashed bytes are parsed bytes
    registry_bytes = registry_path.read_bytes()

    architecture, bundle_seed, model_ids = _check_run_report(checks, report)
    bundle_id = model_ids[SEEDS.index(bundle_seed)]
    _check_claim_v2(checks, report, claim, claim_bytes, model_ids)
    _check_registry(checks, report, claim, registry_path, model_ids)
    manifest, manifest_sha = _load_manifest(checks, Path(manifest_path), risk.get("f2_manifest_sha256"))
    scores = _read_scores(checks, report, run_dir / CLIP_SCORES_DIR, manifest_sha, model_ids)
    _check_same_clips(checks, scores, model_ids)
    subgroups = _f2_join(checks, report, manifest, scores[model_ids[0]])
    policy = _check_evidence(checks, report, model_ids, search_dir, search_bytes, risk, policy_bytes)
    _rederive(checks, policy, scores, bundle_id, model_ids)

    first = scores[model_ids[0]]
    y = np.array([LABEL_CODES[x] for x in first.label], dtype=int)
    groups = list(first.group_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "report_code_commit": commit,
        "architecture": architecture,
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
            "v2_risk_run_report_sha256": _sha256(risk_bytes),
            "v2_search_run_report_sha256": _sha256(search_bytes[SEARCH_RUN_REPORT]),
            "v2_search_selection_report_sha256": _sha256(search_bytes[SEARCH_SELECTION_REPORT]),
            "clip_scores": {m: {k: report["clip_scores"]["files"][m][k] for k in ("csv_sha256", "sidecar_sha256")}
                            for m in model_ids},
        },
        "conventions": {"decision": "D57", "scope": CONVENTIONS_SCOPE, **CONVENTIONS_V2},
        "checks": checks,
        "models": _models(y, subgroups, scores, policy, groups, bundle_id, model_ids),
        "seeds": _seeds(y, scores, model_ids),
        "hypotheses": None,
        "hypotheses_note": HYPOTHESES_NOTE,
    }


def render_markdown(report: dict) -> str:
    """The Markdown report, rendered from the report JSON only (no number computed here)."""
    run = report["run"]
    model_ids = model_ids_v2(report["architecture"])
    models = _ordered(report["models"], model_ids)
    lines = [
        "# Stage 11 FoR test metrics report: model v2 set", "",
        "Generated by `src/evaluation/stage11_v2_report.py` from the stored v2 Stage 11 files only "
        "(no second test run, D31). Every number below comes from `metrics_report.json`; rates are "
        "shown to 4 decimals (full precision in the JSON); `undefined` = denominator 0.", "",
        "**Reported whatever the outcome (pre-registered).**", "",
        f"- Architecture `{report['architecture']}`; run `{run['run_id']}`, run commit `{run['code_commit']}`, "
        f"report commit `{report['report_code_commit']}`",
        f"- Dataset `{run['dataset']}`, split `{run['split']}`, purpose `{run['evaluation_purpose']}`; "
        f"bundle seed {run['bundle_seed']} (`{run['bundle_model_id']}`)",
        f"- {run['n_clips']} clips in {run['n_groups']} D14 groups; labels "
        + ", ".join(f"{k} {v}" for k, v in run["counts"]["label"].items()) + "; subgroups "
        + ", ".join(f"{k} {v}" for k, v in run["counts"]["subgroup"].items()),
        f"- Integrity checks: {sum(c['passed'] for c in report['checks'].values())} of "
        f"{len(report['checks'])} passed; registry SHA-256 `{report['inputs']['registry']['sha256']}`, "
        f"{report['inputs']['registry']['lines']} lines",
        f"- Conventions: {report['conventions']['decision']}, {report['conventions']['scope']} (listed in the "
        "JSON)", "",
        "## Hypotheses", "",
        report["hypotheses_note"], "",
    ]

    lines += ["## Score level (EER and ROC-AUC on raw_score, all three seeds)", "",
              "The threshold at the EER point is descriptive only; the EER is not the error rate at any frozen "
              "threshold (harness Step 1).", ""]
    rows = []
    for model_id, m in models:
        for part, s in _ordered(m["score_level"], SCORE_PARTS):
            rows.append([model_id, part, str(s["n_bonafide"]), str(s["n_spoof"]), _fmt(s["eer"]),
                         _fmt(s["threshold_at_eer_point_descriptive"]), _fmt(s["roc_auc"])])
    lines += _table(["Model", "Rows", "n bonafide", "n spoof", "EER", "Threshold at EER point (descriptive)",
                     "ROC-AUC"], rows)

    lines += ["## Bootstrap 95% CIs (bundle-seed v2 model)", "",
              "Group-clustered (D14), 2,000 resamples, seed 2026. A collapsed interval (low = high) does not "
              "mean there is no uncertainty (harness Step 4).", ""]
    rows = []
    for model_id, m in models:
        for metric, ci in _ordered(m.get("ci", {}), CI_METRICS):
            rows.append([model_id, metric, _fmt(ci["estimate"]), _ci_text(ci), str(ci["n_invalid"]),
                         str(ci["n_groups"]), _counts_text(ci["counts"])])
    lines += _table(["Model", "Metric", "Estimate", "95% CI", "n_invalid", "Groups", "Counts"], rows)

    for model_id, m in models:
        if "forced_binary" not in m:
            continue
        frozen = m["frozen"]
        lines += ["## Forced binary at the frozen threshold", "",
                  f"**{model_id}**: synthetic iff p >= {frozen['threshold_on_p']!r} (policy "
                  f"{frozen['policy_version']}).", ""]
        rows = [[part, *(str(b[k]) for k in ("tp", "fp", "tn", "fn", "n_spoof", "n_bonafide")),
                 *(_fmt(b[k]) for k in ("accuracy", "precision", "recall", "f1", "fpr", "fnr"))]
                for part, b in _ordered(m["forced_binary"], ROW_PARTS)]
        lines += _table(["Rows", "TP", "FP", "TN", "FN", "n spoof", "n bonafide", "Accuracy", "Precision",
                         "Recall", "F1", "FPR", "FNR"], rows)

        lines += ["## Three-way decisions (bundle-seed v2 model, stored predictions)", "",
                  f"Band on p: [{frozen['threshold_on_p']!r} - {frozen['half_width']!r}, "
                  f"{frozen['threshold_on_p']!r} + {frozen['half_width']!r}) is uncertain.", ""]
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
    lines += ["## v2 seeds (score level, separate row; Protocol §5)", "", seeds["note"] + ".", ""]
    rows = [[f"seed {seed}", _fmt(seeds["eer"]["values"][seed]), _fmt(seeds["roc_auc"]["values"][seed])]
            for seed in seeds["eer"]["values"]]
    rows.append(["mean ± SD", f"{_fmt(seeds['eer']['mean'])} ± {_fmt(seeds['eer']['sd'])}",
                 f"{_fmt(seeds['roc_auc']['mean'])} ± {_fmt(seeds['roc_auc']['sd'])}"])
    lines += _table(["Seed", "EER", "ROC-AUC"], rows)
    return "\n".join(lines)


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
    parser.add_argument("--run-dir", required=True, type=Path, help="The folder the v2 Stage 11 score step wrote")
    parser.add_argument("--registry", required=True, type=Path, help="Run registry JSON Lines file")
    parser.add_argument("--manifest", required=True, type=Path, help="Approved F2 FoR manifest")
    parser.add_argument("--search-evidence", required=True, type=Path, help="docs/audit_evidence/v2_search")
    parser.add_argument("--risk-evidence", required=True, type=Path,
                        help="docs/audit_evidence/v2_risk (run_report.json and risk_policy.json)")
    parser.add_argument("--out", required=True, type=Path, help="New output folder")
    args = parser.parse_args(argv)
    if args.out.exists():  # before any work
        raise ValueError(f"output folder already exists: {args.out}")
    report = build_report(run_dir=args.run_dir, registry_path=args.registry, manifest_path=args.manifest,
                          search_evidence_dir=args.search_evidence, risk_evidence_dir=args.risk_evidence)
    json_path, md_path = write_report(report, args.out)
    print(f"{json_path}\n{md_path}")


if __name__ == "__main__":
    main()
