"""Stage 11 FoR test run for the model v2 set (D48 (7); Protocol §7 addendum; D31, D41, D45, D46).

The v2 counterpart of src/evaluation/stage11_run.py, which stays unchanged (it was reviewed and run:
registry lines 1-2). v2 gets its own ("for", "test", "stage11_test") claim for the v2 model set: the
three seeds of the v2 search winner, `<winner>-v2-seed<k>`, k = 0, 1, 2; calibrated p and prediction
for the bundle seed only (D45). No baseline: it was scored in the v1 claim. Everything is scored on
CPU (D46 (1)), through v2_risk_run.model_inputs + v2_risk_run.score_clips, the path risk-v2 was
fitted on. The environment gate and record, the row checks, the feature digests, the bundle-seed
outputs, the exclusive file writes, the check records, the file names and the two-phase procedure
are the v1 runner's own functions, imported, not copied.

Evidence (paths fixed in code, nothing typed):
- V2_SEARCH_EVIDENCE (the constant of v2_risk_run): the winner, bundle_seed (run_report.json and
  selection_report.json agree, v2_risk_run.bundle_checkpoint), checkpoint_sha256 of
  <winner>_seed<k> and best_selection_m of seeds 0, 1, 2; bundle_seed must be select_bundle_seed
  over those M values; val_a_cache_npz_sha256 must be Stage 8's (STAGE8_EVIDENCE).
- V2_RISK_EVIDENCE (docs/audit_evidence/v2_risk/, the risk-v2 run's committed outputs):
  run_report.json (architecture, bundle_seed, checkpoint_sha256 and the hashes of the two search
  reports equal the search evidence; scoring on CPU) and risk_policy.json (SHA-256 = the run
  report's risk_policy_sha256).
- V1_RUN_REPORT (docs/audit_evidence/stage11_run/run_report.json): the v1 test log-Mel hash, recorded
  next to the v2 one, never asserted.

claim() (no test audio is read):
1. Pre-claim checks, nothing written: results_ref and output folder; clean full-SHA commit;
   environment gate (D46 (2)); the evidence above; F3 profile, F2 manifest and
   configs/v2_candidates.yaml hashes and the winner's batch size; each checkpoint hashed and loaded
   against its recorded hash (load_v2_checkpoint), metadata architecture, seed, selection_m,
   S hash and provenance equal to the evidence; the risk policy hash; the reproduction gate:
   log-Mel winners (cnn-aug, crnn): two val-A log-Mel cache builds (cache_a, cache_b),
   byte-identical and equal to Stage 8; aasist: the in-memory val-A F3 waveforms (two passes) equal
   S's clean waveforms (content hash over waveforms, labels, paths). No MFCC and no D58 reference
   (no baseline in this set). Every test row valid under the clip_scores rules. Registry rule
   (D48 (7) with D46 (8)): the ("for", "test", "stage11_test") records are exactly the v1 pair
   (started + completed, model set stage11_run.MODEL_IDS) and no record holds a v2 model ID.
2. RunRegistry.claim for the v2 model set, re-read from the file and verified.
3. out/ with environment.json and pip_freeze.txt, then out/claim.json. The run_id is returned.

score(run_id) (after the notebook has flushed and re-mounted Drive):
4. claim.json must name this run_id and the current commit (no write otherwise); then
   scoring_started.json is created exclusively (no rerun after the claim, D46 (8)) and
   environment_score.json is recorded.
5. The pre-claim checks again; the registry holds the v1 pair and exactly one v2 record: started,
   this run_id, this commit, the v2 model set; registry and out/ still match claim.json.
6. Test inputs in memory, never persisted (D46 (7)): one read_verified_audio + extract_features call
   per clip; the winner's input is kept (log-Mel, or the F3 waveform prepared.waveform for aasist);
   two passes, equal hashes; the hashes are written to features.json before any model sees the
   inputs. Whether the test log-Mel hash equals the v1 run's is recorded, not asserted.
7. Each seed scored twice on CPU, byte-identical; bundle-seed p and prediction from risk-v2, checked
   independently (stage11_run.bundle_outputs).
8. Per-clip score files (roles v2_bundle_seed / v2_other_seed) and run_report.json; no metrics.
9. RunRegistry.complete with the results_ref of claim.json.
Any failure after the claim raises and leaves the started record (D46 (8)).

rehearse() is the dry run on val-A and val-B only (no registry, no test path): (a) the
reproduction gate; (b) per seed, the five S EERs and M recomputed on CPU with train_v2's own
functions must equal best_selection_m exactly (no tolerance; a mismatch stops for a team decision,
as in V4a); (c) val-B: the bundle seed's raw scores and p must equal risk-v2's val_b_scores.npz
bytes (after its SHA-256 check), with the in-memory val-B input equal to the one risk-v2 was fitted
on. The report is always written, "passed" true or false.

Nothing is logged to W&B. This module never caches test inputs, never fits, tunes or recalibrates
anything, and never touches DEEP-VOICE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.evaluation.clip_scores import write_clip_scores
from src.evaluation.run_registry import COMPLETED, STARTED, RunRegistry
from src.evaluation.seeds import select_bundle_seed
from src.evaluation.stage11_run import (
    _COMMIT,
    _RUN_ID,
    CLAIM_JSON,
    CLIP_SCORES_DIR,
    DATASET,
    DEVICE,
    ENVIRONMENT_JSON,
    ENVIRONMENT_SCORE_JSON,
    LABEL_NAMES,
    LOG_MEL,
    PIP_FREEZE,
    PURPOSE,
    RECORDED,
    REHEARSAL_SPLITS,
    RUN_REPORT,
    SCORING_MARKER,
    SPLIT,
    STAGE_REPORT,
    _ArrayDigest,
    _bytes_sha256,
    _check,
    _check_rehearsal_splits,
    _check_results_ref,
    _counts,
    _create,
    _load_cache,
    _load_val_b_scores,
    _read_claim,
    _read_json,
    _record,
    _stage11_records,
    _write_rehearsal_report,
    array_sha256,
    bundle_outputs,
    environment_gate,
    environment_record,
    features_sha256,
    split_rows,
)
from src.evaluation.stage11_run import MODEL_IDS as V1_MODEL_IDS
from src.features.logmel_cache import FEATURE_SHAPE as LOG_MEL_SHAPE
from src.features.mfcc_cache import LABEL_CODES
from src.models.v2_checkpoint import ARCHITECTURES, INPUTS, load_v2_checkpoint, v2_checkpoint_sha256
from src.preprocessing.features import extract_features
from src.preprocessing.profile import load_profile
from src.preprocessing.validation import (
    F2_MANIFEST_SHA256,
    code_version,
    load_f2_manifest,
    read_verified_audio,
    sha256_file,
)
from src.risk.policy import RiskPolicy
from src.training.selection_set import CONDITIONS, load_selection_set
from src.training.train_v2 import evaluate_selection, selection_m, selection_tensors
from src.training.v2_config import load_v2_config
from src.tuning.baseline_run import _json_text
from src.tuning.risk_run import RISK_POLICY
from src.tuning.v2_risk_run import (
    V2_SEARCH_EVIDENCE,
    WAVEFORM_SAMPLES,
    _sha256_arrays,
    bundle_checkpoint,
    model_inputs,
    score_clips,
)
from src.tuning.v2_search import CHECKPOINT_DIR, run_dir_name
from src.tuning.v2_search import RUN_REPORT as SEARCH_RUN_REPORT
from src.tuning.v2_search import SELECTION_REPORT as SEARCH_SELECTION_REPORT

SCHEMA_VERSION = 1
REPO = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPO / "docs" / "audit_evidence"
V2_RISK_EVIDENCE = EVIDENCE_ROOT / "v2_risk"  # committed after the risk-v2 run
STAGE8_EVIDENCE = EVIDENCE_ROOT / "stage8_cnn"
V1_RUN_REPORT = EVIDENCE_ROOT / "stage11_run" / RUN_REPORT
SEEDS = (0, 1, 2)  # D11, D49 final seeds
ROLE_BUNDLE = "v2_bundle_seed"
ROLE_OTHER = "v2_other_seed"
WAVEFORM = "waveform"  # the F3 waveform input (aasist); INPUTS value and feature name
GATE_SPLIT = "val-A"
FIT_SPLIT = "val-B"
FEATURES_JSON = "features.json"
V1_MODEL_SET = tuple(sorted(V1_MODEL_IDS))


def model_ids_v2(winner: str) -> tuple[str, ...]:
    """The v2 model set (D48 (7)): `<winner>-v2-seed<k>` for k = 0, 1, 2, in seed order."""
    if winner not in ARCHITECTURES:
        raise ValueError(f"winner must be one of {ARCHITECTURES}, got {winner!r}")
    return tuple(f"{winner}-v2-seed{k}" for k in SEEDS)


# Every possible v2 model ID: a registry record holding any of them is a v2 record.
ALL_V2_MODEL_IDS = frozenset(m for a in ARCHITECTURES for m in model_ids_v2(a))


def model_roles(winner: str, bundle_seed: int) -> dict[str, str]:
    """{model_id: role}: v2_bundle_seed for the bundle seed (D45), v2_other_seed for the others."""
    if bundle_seed not in SEEDS:
        raise ValueError(f"bundle_seed must be one of {SEEDS}, got {bundle_seed!r}")
    return {m: ROLE_BUNDLE if k == bundle_seed else ROLE_OTHER for k, m in zip(SEEDS, model_ids_v2(winner))}


def _require(field: str, expected, actual) -> None:
    """Refuse a disagreement, naming the field, the expected and the actual value."""
    same = isinstance(actual, bool) == isinstance(expected, bool) and actual == expected
    if not same:
        raise ValueError(f"evidence check failed: {field}: expected {expected!r}, got {actual!r}")


# --- Evidence -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class V2Evidence:
    """The expected values, read from the committed v2 search, risk-v2, Stage 8 and v1 Stage 11 files."""

    winner: str
    bundle_seed: int
    checkpoint_sha256: dict  # seed (int) -> combined checkpoint hash
    best_selection_m: dict  # seed (int) -> M
    selection_set_sha256: str
    v2_config_sha256: str
    profile_fingerprint: str
    profile_config_sha256: str
    f2_manifest_sha256: str
    val_a_cache_npz_sha256: str  # Stage 8 val-A log-Mel
    risk_policy_sha256: str
    val_b_scores_sha256: str
    val_b_input: dict  # risk-v2's val_b_input record
    batch_size: int  # risk-v2 scoring batch size
    v1_test_log_mel_sha256: str  # recorded next to v2's, never asserted
    v1_run_report_sha256: str


def _number_in_unit(field: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"evidence check failed: {field}: expected a finite number in [0, 1], got {value!r}")
    return value


def load_v2_evidence() -> V2Evidence:
    """Read and cross-check the committed evidence (module docstring); raise ValueError on any disagreement."""
    search = Path(V2_SEARCH_EVIDENCE)
    winner, bundle_seed, bundle_sha = bundle_checkpoint(search)  # run_report and selection_report agree
    run_report = _read_json(search / SEARCH_RUN_REPORT)
    selection = _read_json(search / SEARCH_SELECTION_REPORT)
    hashes = run_report["checkpoint_sha256"]
    names = {k: run_dir_name(winner, k) for k in SEEDS}
    _require("v2 search run_report checkpoint_sha256 entries for the winner",
             sorted(names.values()), sorted(n for n in hashes if n in names.values()))
    seeds = selection["seeds"]
    _require("v2 search selection_report seeds", sorted(str(k) for k in SEEDS), sorted(seeds))
    ms = {k: _number_in_unit(f"v2 search selection_report seeds.{k}.best_selection_m",
                             seeds[str(k)]["best_selection_m"]) for k in SEEDS}
    _require("v2 search bundle_seed (select_bundle_seed over seeds.*.best_selection_m)",
             select_bundle_seed(ms), bundle_seed)
    stage8 = _read_json(Path(STAGE8_EVIDENCE) / STAGE_REPORT)
    _require("v2 search run_report val_a_cache_npz_sha256 (Stage 8 val_a_cache_npz_sha256)",
             stage8["val_a_cache_npz_sha256"], run_report["val_a_cache_npz_sha256"])

    risk_dir = Path(V2_RISK_EVIDENCE)
    risk = _read_json(risk_dir / RUN_REPORT)
    _require("v2 risk run_report architecture", winner, risk["architecture"])
    _require("v2 risk run_report bundle_seed", bundle_seed, risk["bundle_seed"])
    _require("v2 risk run_report checkpoint_sha256", bundle_sha, risk["checkpoint_sha256"])
    _require("v2 risk run_report scoring.device", DEVICE, risk["scoring"]["device"])
    _require("v2 risk run_report val_b_input.form", INPUTS[winner], risk["val_b_input"]["form"])
    for key in ("profile_fingerprint", "profile_config_sha256", "f2_manifest_sha256"):
        _require(f"v2 risk run_report {key} (v2 search run_report {key})", run_report[key], risk[key])
    _require("v2 risk run_report v2_search_evidence (SHA-256 of the committed search reports)",
             {name: sha256_file(search / name) for name in (SEARCH_RUN_REPORT, SEARCH_SELECTION_REPORT)},
             risk["v2_search_evidence"])
    v1 = _read_json(Path(V1_RUN_REPORT))
    return V2Evidence(
        winner=winner,
        bundle_seed=bundle_seed,
        checkpoint_sha256={k: hashes[names[k]] for k in SEEDS},
        best_selection_m=ms,
        selection_set_sha256=run_report["selection_set_sha256"],
        v2_config_sha256=run_report["v2_config_sha256"],
        profile_fingerprint=run_report["profile_fingerprint"],
        profile_config_sha256=run_report["profile_config_sha256"],
        f2_manifest_sha256=run_report["f2_manifest_sha256"],
        val_a_cache_npz_sha256=run_report["val_a_cache_npz_sha256"],
        risk_policy_sha256=risk["risk_policy_sha256"],
        val_b_scores_sha256=risk["val_b_scores_sha256"],
        val_b_input=dict(risk["val_b_input"]),
        batch_size=risk["scoring"]["batch_size"],
        v1_test_log_mel_sha256=v1["features"]["arrays"][LOG_MEL],
        v1_run_report_sha256=sha256_file(Path(V1_RUN_REPORT)),
    )


# --- Inputs (in memory, never persisted) ----------------------------------------------------------

def _rows_arrays(rows) -> tuple[np.ndarray, np.ndarray]:
    """(labels int8, 1 = FAKE; paths unicode) of rows, the forms S and risk-v2 hash."""
    return (rows["label"].map(LABEL_CODES).to_numpy(dtype=np.int8), np.array(rows["path"].tolist()))


def _input_pass(root, rows, profile, kind: str, *, keep: bool):
    """One pass over rows: per clip one read_verified_audio + extract_features call.

    The log-Mel is digested for every winner (stage11_run._ArrayDigest, the v1 definition, so it can
    be compared with the v1 run); aasist's F3 waveforms are hashed with v2_risk_run._sha256_arrays.
    Returns (the winner's input array if keep else None, hash record).
    """
    n = len(rows)
    log_mel_digest = _ArrayDigest(LOG_MEL, n)
    log_mels = np.empty((n, *LOG_MEL_SHAPE), dtype=np.float32) if kind == "logmel" and keep else None
    waves = np.empty((n, WAVEFORM_SAMPLES), dtype=np.float32) if kind == WAVEFORM else None
    for i, (_, row) in enumerate(rows.iterrows()):
        waveform, sample_rate = read_verified_audio(root, row)
        prepared, log_mel, _ = extract_features(waveform, sample_rate, profile)
        log_mel_digest.update(log_mel)
        if log_mels is not None:
            log_mels[i] = log_mel
        if waves is not None:
            w = np.asarray(prepared.waveform)
            if w.dtype != np.float32 or w.shape != (WAVEFORM_SAMPLES,) or not np.isfinite(w).all():
                raise ValueError(f"F3 waveform must be finite float32 ({WAVEFORM_SAMPLES},), got {w.dtype} {w.shape}")
            waves[i] = w
    arrays = {LOG_MEL: log_mel_digest.hexdigest()}
    if waves is not None:
        arrays[WAVEFORM] = _sha256_arrays(waves)
    record = {"arrays": arrays,
              "features_sha256": features_sha256(rows["path"].tolist(), rows["label"].tolist(), arrays)}
    kept = (log_mels if kind == "logmel" else waves) if keep else None
    return kept, record


def compute_inputs(root, rows, profile, kind: str) -> tuple[np.ndarray, dict[str, Any]]:
    """The winner's input in memory, computed twice (D46 (7)); raise if the two passes' hashes differ."""
    x, first = _input_pass(root, rows, profile, kind, keep=True)
    _, second = _input_pass(root, rows, profile, kind, keep=False)
    if first != second:
        raise ValueError(f"input passes differ: pass 1 {first}, pass 2 {second}")
    return x, {**first, "input": kind, "n_clips": len(rows)}


# --- Models and scoring ---------------------------------------------------------------------------

@dataclass
class Context:
    commit: str
    evidence: V2Evidence
    profile: Any
    manifest: Any
    models: dict  # seed -> (model, mean | None, std | None, metadata)
    policy: RiskPolicy
    batch_size: int
    checks: dict
    profile_config_sha256: str
    f2_manifest_sha256: str

    @property
    def kind(self) -> str:
        return INPUTS[self.evidence.winner]

    @property
    def model_ids(self) -> tuple[str, ...]:
        return model_ids_v2(self.evidence.winner)


def score_models(ctx: Context, x: np.ndarray, seeds=SEEDS) -> dict[str, np.ndarray]:
    """Raw float64 scores per model id on CPU through risk-v2's path, each twice; raise unless byte-identical."""
    out = {}
    for seed in seeds:
        model_id = ctx.model_ids[SEEDS.index(seed)]
        model, mean, std, _ = ctx.models[seed]
        tensor = model_inputs(ctx.kind, x, mean, std)
        first, second = score_clips(model, tensor, ctx.batch_size), score_clips(model, tensor, ctx.batch_size)
        if first.dtype != np.float64 or first.tobytes() != second.tobytes():
            raise ValueError(f"{model_id}: scores differ between two CPU scoring passes")
        out[model_id] = first
    return out


def _model_set(ctx: Context) -> dict[str, dict]:
    roles = model_roles(ctx.evidence.winner, ctx.evidence.bundle_seed)
    return {m: {"role": roles[m], "artifact_sha256": ctx.evidence.checkpoint_sha256[k]}
            for k, m in zip(SEEDS, ctx.model_ids)}


# --- Pre-claim checks (shared by claim, score and rehearse) ---------------------------------------

def _check_input_kind(winner: str, cache_a, cache_b) -> None:
    if INPUTS[winner] == "logmel":
        if cache_a is None or cache_b is None:
            raise ValueError(f"{winner} takes the log-Mel: cache_a and cache_b (two val-A log-Mel cache "
                             f"builds) are required")
    elif cache_a is not None or cache_b is not None:
        raise ValueError(f"{winner} takes the F3 waveform: cache_a and cache_b must not be given (its "
                         f"gate compares the in-memory val-A waveforms with S)")


def _load_models(checkpoints, ev: V2Evidence, checks: dict) -> dict:
    models = {}
    for seed in SEEDS:
        name = run_dir_name(ev.winner, seed)
        folder, expected, prefix = Path(checkpoints) / name / CHECKPOINT_DIR, ev.checkpoint_sha256[seed], \
            f"checkpoint.{name}"
        # The repository's hash (the cnn_checkpoint rule, as load_v2_checkpoint computes it), then
        # load_v2_checkpoint checks the bytes it loads against expected again.
        _check(checks, f"{prefix}.sha256", expected, v2_checkpoint_sha256(folder))
        loaded = load_v2_checkpoint(folder, expected_sha256=expected)
        meta = loaded[3]
        _check(checks, f"{prefix}.architecture", ev.winner, meta["architecture"])
        _check(checks, f"{prefix}.seed", seed, meta["seed"])
        _check(checks, f"{prefix}.selection_m", ev.best_selection_m[seed], meta["selection_m"])
        for key in ("selection_set_sha256", "profile_fingerprint", "f2_manifest_sha256"):
            _check(checks, f"{prefix}.{key}", getattr(ev, key), meta[key])
        models[seed] = loaded
    return models


def _gate_selection_set(selection_set, profile, manifest, ev: V2Evidence, checks: dict, key: str, **kw):
    selection = load_selection_set(selection_set, profile, form=INPUTS[ev.winner], f2_manifest=manifest, **kw)
    _check(checks, key, ev.selection_set_sha256, selection.sha256)
    return selection


def reproduction_gate(checks: dict, *, root, cache_a, cache_b, selection_set, profile, profile_config_sha256,
                      manifest, ev: V2Evidence) -> None:
    """Before any test access: log-Mel winners rebuild val-A twice (byte-identical, equal to Stage 8);
    aasist's in-memory val-A F3 waveforms (two passes) must equal S's clean waveforms."""
    if INPUTS[ev.winner] == "logmel":
        if Path(cache_a).resolve() == Path(cache_b).resolve():
            raise ValueError(f"cache_a and cache_b are the same folder ({cache_a}); the gate needs two builds")
        prefix = f"reproduction.{LOG_MEL}.{GATE_SPLIT}"
        builds = [_load_cache(c, LOG_MEL, GATE_SPLIT, profile, profile_config_sha256) for c in (cache_a, cache_b)]
        observed = builds[0][3]["npz_sha256"]
        _check(checks, f"{prefix}.builds_identical", observed, builds[1][3]["npz_sha256"])
        _check(checks, f"{prefix}.npz_sha256", ev.val_a_cache_npz_sha256, observed)
        return
    prefix = f"reproduction.{WAVEFORM}.{GATE_SPLIT}"
    selection = _gate_selection_set(selection_set, profile, manifest, ev, checks, f"{prefix}.selection_set_sha256")
    expected = _sha256_arrays(selection.features("clean"), selection.labels, selection.paths)
    del selection
    rows = split_rows(manifest, GATE_SPLIT)
    waves, _ = compute_inputs(root, rows, profile, WAVEFORM)
    _check(checks, f"{prefix}.equals_selection_clean", expected, _sha256_arrays(waves, *_rows_arrays(rows)))


def preflight(checks: dict, *, root, checkpoints, profile_path, config_path, manifest_path, selection_set,
              cache_a=None, cache_b=None) -> Context:
    """Every check that reads no test audio and writes nothing; raise on the first failure.

    Checks are recorded into the caller's dict as they run (the rehearsal report keeps them).
    """
    commit = code_version()["git_commit"]  # refuses a dirty worktree
    _record(checks, "code_commit", "full 40-hex SHA", commit,
            isinstance(commit, str) and _COMMIT.fullmatch(commit) is not None)
    environment_gate(checks)
    ev = load_v2_evidence()
    _check_input_kind(ev.winner, cache_a, cache_b)

    profile_path, config_path, manifest_path = Path(profile_path), Path(config_path), Path(manifest_path)
    profile = load_profile(profile_path)
    profile_config_sha256 = sha256_file(profile_path)
    f2_manifest_sha256 = sha256_file(manifest_path)
    _check(checks, "profile_fingerprint", ev.profile_fingerprint, profile.fingerprint())
    _check(checks, "profile_config_sha256", ev.profile_config_sha256, profile_config_sha256)
    _check(checks, "f2_manifest_sha256", ev.f2_manifest_sha256, f2_manifest_sha256)
    _check(checks, "f2_manifest_sha256_approved", F2_MANIFEST_SHA256, f2_manifest_sha256)
    manifest = load_f2_manifest(manifest_path)
    _check(checks, "v2_config_sha256", ev.v2_config_sha256, sha256_file(config_path))
    batch_size = load_v2_config(config_path).require_batch_sizes()[ev.winner]
    _check(checks, "batch_size", ev.batch_size, batch_size)

    models = _load_models(checkpoints, ev, checks)
    data = (Path(V2_RISK_EVIDENCE) / RISK_POLICY).read_bytes()  # one read: hashed bytes are parsed bytes
    _check(checks, "risk_policy.sha256", ev.risk_policy_sha256, hashlib.sha256(data).hexdigest())
    policy = RiskPolicy.from_json(data.decode("utf-8"))
    reproduction_gate(checks, root=root, cache_a=cache_a, cache_b=cache_b, selection_set=selection_set,
                      profile=profile, profile_config_sha256=profile_config_sha256, manifest=manifest, ev=ev)
    return Context(commit=commit, evidence=ev, profile=profile, manifest=manifest, models=models, policy=policy,
                   batch_size=batch_size, checks=checks, profile_config_sha256=profile_config_sha256,
                   f2_manifest_sha256=f2_manifest_sha256)


# --- Registry rule (D48 (7) with D46 (8)) ---------------------------------------------------------

def _describe(records) -> list:
    return [(r.run_id, r.status, r.evaluation_purpose, list(r.model_set)) for r in records]


def _v2_records(registry: RunRegistry) -> list:
    return [r for r in registry.records() if ALL_V2_MODEL_IDS & set(r.model_set)]


def _is_v1_pair(records) -> bool:
    return ([r.status for r in records] == [STARTED, COMPLETED] and len({r.run_id for r in records}) == 1
            and all(r.model_set == V1_MODEL_SET for r in records))


def check_registry_before_claim(registry: RunRegistry) -> list:
    """The stage11_test records must be exactly the v1 pair, and no record may hold a v2 model ID."""
    v2 = _v2_records(registry)
    if v2:
        raise ValueError(f"the registry already holds records with v2 model IDs: {_describe(v2)}; a second "
                         f"v2 claim needs a new decision (D31, D46 (8))")
    stage11 = _stage11_records(registry)
    if not _is_v1_pair(stage11):
        raise ValueError(f"before the v2 claim the {PURPOSE} records must be exactly the v1 pair (started + "
                         f"completed, model set {list(V1_MODEL_SET)}), found {_describe(stage11)}")
    return _describe(stage11)


def check_registry_after_claim(registry: RunRegistry, run_id: str, commit: str, model_ids) -> None:
    """The v1 pair, plus exactly one v2 record: started, this run_id, this commit, this model set."""
    stage11 = _stage11_records(registry)
    v1 = [r for r in stage11 if r.model_set == V1_MODEL_SET]
    v2 = _v2_records(registry)
    others = [r for r in stage11 if r not in v1 and r not in v2]
    wanted = (run_id, STARTED, commit, PURPOSE, tuple(sorted(model_ids)))
    if not _is_v1_pair(v1) or others or len(v2) != 1 or \
            (v2[0].run_id, v2[0].status, v2[0].code_commit, v2[0].evaluation_purpose, v2[0].model_set) != wanted:
        raise ValueError(f"the registry must hold the v1 pair and exactly one started v2 {PURPOSE} record for "
                         f"run_id {run_id} at commit {commit}, found {_describe(stage11 + [r for r in v2 if r not in stage11])}")


# --- Official run, phase 1: claim ---------------------------------------------------------------

def claim(*, root, registry_path, out_dir, results_ref, **inputs) -> dict[str, Any]:
    """Phase 1 (module docstring, steps 1-3): check, claim, write the evidence; return claim.json.

    inputs are preflight()'s keyword arguments. No test audio is read (aasist's gate reads val-A).
    """
    results_ref = _check_results_ref(results_ref)
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError(f"output folder already exists: {out_dir}")
    ctx = preflight({}, root=root, **inputs)
    counts = _counts(split_rows(ctx.manifest, SPLIT))  # manifest metadata only; no audio is read
    registry = RunRegistry(registry_path)  # an invalid or edited file fails here
    found = check_registry_before_claim(registry)
    ctx.checks["registry.v1_pair_only"] = {"expected": f"the v1 pair {list(V1_MODEL_SET)}, no v2 record",
                                           "actual": found, "passed": True}
    environment, freeze = environment_record()  # recorded, not asserted

    model_ids = ctx.model_ids
    run_id = registry.claim(DATASET, SPLIT, model_ids, ctx.commit, evaluation_purpose=PURPOSE)
    last = RunRegistry(registry_path).records()[-1]  # re-read the claim from the file
    if (last.run_id, last.status, last.code_commit) != (run_id, STARTED, ctx.commit) \
            or last.key != (DATASET, SPLIT, PURPOSE, tuple(sorted(model_ids))):
        raise ValueError(f"the registry's last record is not this run's claim {run_id}: {last}")

    # From here on, a failure raises and the started record stays (D46 (8)).
    out_dir.mkdir(parents=True)  # exclusive
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
        "architecture": ctx.evidence.winner,
        "bundle_seed": ctx.evidence.bundle_seed,
        "model_ids": sorted(model_ids),
        "pre_claim_checks": ctx.checks,
        "environment_files": {ENVIRONMENT_JSON: hashlib.sha256(environment_bytes).hexdigest(),
                              PIP_FREEZE: hashlib.sha256(freeze_bytes).hexdigest()},
        "counts": counts,
    }
    _create(out_dir / CLAIM_JSON, _json_text(record).encode("utf-8"))
    return record


# --- Official run, phase 2: score -----------------------------------------------------------------

def score(*, root, registry_path, out_dir, run_id, **inputs) -> dict[str, Any]:
    """Phase 2 (module docstring, steps 4-9): score the claimed run once; return run_report.json."""
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
    environment_score_bytes = _json_text(environment_record()[0]).encode("utf-8")
    _create(out_dir / ENVIRONMENT_SCORE_JSON, environment_score_bytes)

    ctx = preflight({}, root=root, **inputs)
    rows = split_rows(ctx.manifest, SPLIT)
    check_registry_after_claim(RunRegistry(registry_path), run_id, ctx.commit, ctx.model_ids)
    claim_record, claim_sha256 = _read_claim(out_dir, run_id, ctx.commit, registry_path)
    if _counts(rows) != claim_record["counts"]:
        raise ValueError(f"test rows changed since the claim: {_counts(rows)} != {claim_record['counts']}")

    x, feature_hashes = compute_inputs(root, rows, ctx.profile, ctx.kind)
    ev = ctx.evidence
    test_log_mel = feature_hashes["arrays"][LOG_MEL]
    features_record = {
        "features": feature_hashes,
        # Recorded, not asserted: a mismatch after the claim cannot be undone; it is for team review.
        "v1_log_mel_comparison": {"v1_run_report_sha256": ev.v1_run_report_sha256,
                                  "v1_log_mel_sha256": ev.v1_test_log_mel_sha256,
                                  "v2_log_mel_sha256": test_log_mel,
                                  "equal": test_log_mel == ev.v1_test_log_mel_sha256},
    }
    features_bytes = _json_text(features_record).encode("utf-8")
    _create(out_dir / FEATURES_JSON, features_bytes)  # recorded before any model sees the inputs (D46 (7))

    scores = score_models(ctx, x)
    del x
    model_set = _model_set(ctx)
    bundle_id = ctx.model_ids[SEEDS.index(ev.bundle_seed)]
    p, prediction = bundle_outputs(ctx.policy, scores[bundle_id])

    clip_dir = out_dir / CLIP_SCORES_DIR
    clip_dir.mkdir()  # exclusive
    columns = {"clip_id": rows["path"].tolist(), "label": rows["label"].tolist(),
               "group_id": rows["group"].tolist()}
    clip_files = {}
    for model_id in ctx.model_ids:
        extra = {"p": p, "prediction": prediction} if model_id == bundle_id else {}
        csv_path, json_path = write_clip_scores(
            clip_dir, evaluation_purpose=PURPOSE, run_id=run_id, model_id=model_id,
            model_role=model_set[model_id]["role"], model_artifact_sha256=model_set[model_id]["artifact_sha256"],
            manifest_sha256=ctx.f2_manifest_sha256, raw_score=scores[model_id], **columns, **extra,
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
        "architecture": ev.winner,
        "input": ctx.kind,
        "model_set": model_set,
        "bundle_seed": ev.bundle_seed,
        "claim_sha256": claim_sha256,
        "scoring_marker_sha256": sha256_file(out_dir / SCORING_MARKER),
        "features_json_sha256": hashlib.sha256(features_bytes).hexdigest(),
        "pre_scoring_checks": ctx.checks,
        "environment_files": claim_record["environment_files"],
        "environment_score_sha256": hashlib.sha256(environment_score_bytes).hexdigest(),
        "environment_same_as_claim": environment_score_bytes == (out_dir / ENVIRONMENT_JSON).read_bytes(),
        "scoring": {"device": DEVICE, "batch_size": ctx.batch_size, "passes": 2},
        **features_record,
        "clip_scores": {"folder": CLIP_SCORES_DIR, "files": clip_files},
        "counts": claim_record["counts"],
    }
    _create(out_dir / RUN_REPORT, _json_text(report).encode("utf-8"))
    RunRegistry(registry_path).complete(run_id, results_ref=claim_record["results_ref"])
    return report


# --- Dry run (validation splits only) -------------------------------------------------------------

def _load_risk_scores(path, expected_sha256: str, known: dict) -> dict[str, np.ndarray]:
    """risk-v2's val_b_scores.npz after its SHA-256 check (recorded here); parsed by stage11_run's loader."""
    _check(known, "risk_v2.val_b_scores_sha256", expected_sha256, sha256_file(Path(path)))
    return _load_val_b_scores(path, expected_sha256, {})  # re-checks the bytes it parses


def rehearse(*, root, risk_scores, out_dir, splits=REHEARSAL_SPLITS, **inputs) -> dict[str, Any]:
    """The dry run on val-A (through S) and val-B (module docstring); return the rehearsal report.

    For log-Mel winners cache_a must also hold log_mel/val-B (the cache risk-v2 was fitted on).
    No registry is opened and no test path exists here. After the split and folder checks the report
    is always written ("passed" true or false, the failing check and the error); then a failure is
    raised again.
    """
    splits = _check_rehearsal_splits(splits)  # before anything is read
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError(f"output folder already exists: {out_dir}")
    environment, freeze = environment_record()
    checks: dict = {}
    known: dict = {}
    report = {
        "schema_version": SCHEMA_VERSION,
        "code_commit": None,
        "splits": list(splits),
        "pre_claim_checks": checks,
        "known_answers": known,
        "environment": {**environment, "pip_freeze_file": PIP_FREEZE},
    }
    try:
        ctx = preflight(checks, root=root, **inputs)
        ev = ctx.evidence
        report.update({"code_commit": ctx.commit, "architecture": ev.winner, "input": ctx.kind,
                       "model_set": _model_set(ctx), "bundle_seed": ev.bundle_seed,
                       "scoring": {"device": DEVICE, "batch_size": ctx.batch_size, "passes": 2}})

        # (b) val-A through S: M per seed by train_v2's own functions, exactly (no tolerance).
        clean = None
        if ctx.kind == "logmel":  # S's clean log-Mel must equal the verified Stage 8 val-A cache
            clean = _load_cache(inputs["cache_a"], LOG_MEL, GATE_SPLIT, ctx.profile, ctx.profile_config_sha256)
        selection = _gate_selection_set(inputs["selection_set"], ctx.profile, ctx.manifest, ev, known,
                                        f"{GATE_SPLIT}.selection_set_sha256", clean_reference_logmel=clean)
        del clean
        for seed, model_id in zip(SEEDS, ctx.model_ids):
            model, mean, std, _ = ctx.models[seed]
            tensors = selection_tensors(selection, ctx.kind, mean, std)
            eers, _ = evaluate_selection(model, tensors, selection.labels, ctx.batch_size, torch.device(DEVICE))
            del tensors
            _record(known, f"{GATE_SPLIT}.{model_id}.selection_eers", RECORDED, {c: eers[c] for c in CONDITIONS},
                    True)
            _check(known, f"{GATE_SPLIT}.{model_id}.selection_m", ev.best_selection_m[seed], selection_m(eers))
        del selection

        # (c) val-B: the bundle seed (D45) against risk-v2's stored scores and p, byte for byte.
        rows = split_rows(ctx.manifest, FIT_SPLIT)
        x, feature_hashes = compute_inputs(root, rows, ctx.profile, ctx.kind)
        labels, paths = _rows_arrays(rows)
        if ctx.kind == "logmel":
            cached, c_labels, c_paths, meta = _load_cache(inputs["cache_a"], LOG_MEL, FIT_SPLIT, ctx.profile,
                                                          ctx.profile_config_sha256)
            _check(known, f"{FIT_SPLIT}.{LOG_MEL}.cache_npz_sha256", ev.val_b_input["npz_sha256"], meta["npz_sha256"])
            cache_hash = features_sha256(c_paths.tolist(), [LABEL_NAMES[int(v)] for v in c_labels],
                                         {LOG_MEL: array_sha256(LOG_MEL, cached)})
            del cached
            memory_hash = features_sha256(rows["path"].tolist(), rows["label"].tolist(),
                                          {LOG_MEL: feature_hashes["arrays"][LOG_MEL]})
            _check(known, f"{FIT_SPLIT}.{LOG_MEL}.in_memory_equals_cache", cache_hash, memory_hash)
        else:
            _check(known, f"{FIT_SPLIT}.{WAVEFORM}.content_sha256", ev.val_b_input["content_sha256"],
                   _sha256_arrays(x, labels, paths))
        bundle_id = ctx.model_ids[SEEDS.index(ev.bundle_seed)]
        scores = score_models(ctx, x, seeds=(ev.bundle_seed,))
        del x
        stored = _load_risk_scores(risk_scores, ev.val_b_scores_sha256, known)
        p, _ = bundle_outputs(ctx.policy, scores[bundle_id])
        _check(known, f"{FIT_SPLIT}.risk_v2.paths", paths.tolist(), stored["paths"].tolist())
        _check(known, f"{FIT_SPLIT}.risk_v2.labels", labels.astype(int).tolist(), stored["labels"].astype(int).tolist())
        _check(known, f"{FIT_SPLIT}.{bundle_id}.raw_score_bytes", _bytes_sha256(stored["score"]),
               _bytes_sha256(scores[bundle_id]))
        _check(known, f"{FIT_SPLIT}.{bundle_id}.p_bytes", _bytes_sha256(stored["p"]), _bytes_sha256(p))
        report["per_split"] = {FIT_SPLIT: {"features": feature_hashes, "scored": sorted(scores),
                                           "counts": _counts(rows)}}
    except Exception as error:
        if report["code_commit"] is None and "code_commit" in checks:  # preflight failed after reading it
            report["code_commit"] = checks["code_commit"]["actual"]
        report.update({"passed": False, "failed_check": getattr(error, "key", None),
                       "error": f"{type(error).__name__}: {error}"})
        _write_rehearsal_report(out_dir, report, freeze)
        raise
    passed = all(c["passed"] for c in (*checks.values(), *known.values()))
    report.update({"passed": passed, "failed_check": None, "error": None})
    _write_rehearsal_report(out_dir, report, freeze)
    if not passed:
        raise ValueError("a rehearsal check is recorded as failed")
    return report


# --- CLI ------------------------------------------------------------------------------------------

def _inputs(args) -> dict[str, Any]:
    return {
        "checkpoints": args.checkpoints, "profile_path": args.profile, "config_path": args.config,
        "manifest_path": args.manifest, "selection_set": args.selection_set, "cache_a": args.cache_a,
        "cache_b": args.cache_b,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("rehearse", "dry run on val-A (through S) and val-B (no registry)"),
                       ("claim", "official run, phase 1: checks, registry claim, evidence (no test audio)"),
                       ("score", "official run, phase 2: score the claimed run once")):
        p = commands.add_parser(name, help=text)
        p.add_argument("--root", required=True, type=Path,
                       help="Local extracted FoR root (claim reads val-A only, for an aasist gate)")
        p.add_argument("--checkpoints", required=True, type=Path,
                       help="The v2 search runs/ folder: <winner>_seed<k>/checkpoint/ for k = 0, 1, 2")
        p.add_argument("--selection-set", required=True, type=Path, help="Selection set S folder")
        p.add_argument("--profile", required=True, type=Path, help="F3 preprocessing profile YAML")
        p.add_argument("--config", required=True, type=Path, help="configs/v2_candidates.yaml (batch size)")
        p.add_argument("--manifest", required=True, type=Path, help="Approved F2 FoR manifest")
        p.add_argument("--cache-a", type=Path, default=None,
                       help="log-Mel winners only: build A, log_mel/val-A/ (and log_mel/val-B/ for rehearse)")
        p.add_argument("--cache-b", type=Path, default=None, help="log-Mel winners only: build B, log_mel/val-A/")
        p.add_argument("--out", required=True, type=Path,
                       help="rehearse, claim: new output folder; score: the folder claim wrote")
        if name == "rehearse":
            p.add_argument("--risk-scores", required=True, type=Path, help="risk-v2 val_b_scores.npz")
        else:
            p.add_argument("--registry", required=True, type=Path, help="Run registry JSON Lines file")
        if name == "claim":
            p.add_argument("--results-ref", required=True,
                           help="Relative POSIX path complete() records, e.g. stage11_v2/run_report.json")
        if name == "score":
            p.add_argument("--run-id", required=True, help="The run_id printed by claim")
    args = parser.parse_args(argv)
    if args.command == "claim":
        report = claim(root=args.root, registry_path=args.registry, out_dir=args.out,
                       results_ref=args.results_ref, **_inputs(args))
    elif args.command == "score":
        report = score(root=args.root, registry_path=args.registry, out_dir=args.out, run_id=args.run_id,
                       **_inputs(args))
    else:
        report = rehearse(root=args.root, risk_scores=args.risk_scores, out_dir=args.out, **_inputs(args))
    print(_json_text(report), end="")


if __name__ == "__main__":
    main()
