"""Model v2 risk-policy run (Protocol v1.0 §6 unchanged; D2, D9, D12, D44, D45, D49, D61): val-B fit.

The v2 counterpart of src/tuning/risk_run.py (Stage 10), which stays v1-only. Steps:
1. Bundle checkpoint from the committed v2 search evidence (V2_SEARCH_EVIDENCE, fixed here): the
   winner (`selected`) and `bundle_seed` must agree between run_report.json and
   selection_report.json; the expected hash is checkpoint_sha256[run_dir_name(winner, seed)].
   load_v2_checkpoint verifies it; the metadata must name the winner and the bundle seed (D45).
2. Provenance checks ({expected, actual}): profile_fingerprint, f2_manifest_sha256,
   selection_set_sha256.
3. Reproduction check (verification only, no decision): the five S EERs and M recomputed on CPU
   with train_v2's own functions (selection_tensors, evaluate_selection, selection_m) must give the
   checkpoint's selection_m exactly; there is no tolerance (that would be a team decision).
4. val-B scored on CPU by architecture: cnn-aug / crnn from the Stage 8 val-B log-Mel cache
   standardized with the checkpoint's mean / std; aasist from an F3 waveform val-B source passed in
   as a function (src/tuning/v2_val_b_source.py, PR V4c: val-B through F3 exactly as
   stage11_v2_run reads it; the CLI builds it from --root). Both through train_v2's raw-logit path;
   two passes must give equal scores.
5. fit_risk_policy unchanged; the refit must be byte-identical. Same files and formats as Stage 10:
   risk_policy.json, val_b_scores.npz, diagnostics_report.json, run_report.json.

Only val-A (through S) and val-B are read: there is no split argument and no other split or
external path (D9, D31). Nothing is logged to W&B (D40). out_dir must not exist; on any failure
it is removed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.models.baseline import _write_npz
from src.models.v2_checkpoint import ARCHITECTURES, INPUT_SHAPES, INPUTS, load_v2_checkpoint
from src.preprocessing.profile import load_profile
from src.preprocessing.validation import code_version, load_f2_manifest, sha256_file
from src.training.selection_set import CONDITIONS, load_selection_set
from src.training.train_cnn import _score
from src.training.train_v2 import evaluate_selection, selection_m, selection_tensors
from src.training.normalization import apply_bin_stats
from src.training.v2_config import load_v2_config
from src.tuning.baseline_run import _json_text
from src.tuning.risk_fit import fit_risk_policy
from src.tuning.risk_run import (
    DEVICE,
    DIAGNOSTICS_REPORT,
    FIT_SPLIT,
    RISK_POLICY,
    RUN_REPORT,
    VAL_B_SCORES,
    _load_split,
    _split_rows,
    diagnostics,
    environment,
)
from src.tuning.v2_search import RUN_REPORT as SEARCH_RUN_REPORT
from src.tuning.v2_search import SELECTION_REPORT as SEARCH_SELECTION_REPORT
from src.tuning.v2_search import run_dir_name

SCHEMA_VERSION = 1
REPO = Path(__file__).resolve().parents[2]
V2_SEARCH_EVIDENCE = REPO / "docs" / "audit_evidence" / "v2_search"  # committed after the search
WAVEFORM_SAMPLES = INPUT_SHAPES["waveform"][0]

# An F3 waveform val-B source for aasist: () -> (waveforms float32 (n, 32000), labels int8 (n,),
# paths unicode (n,), sorted and unique). Built by src/tuning/v2_val_b_source.py (PR V4c).
ValBWaveformSource = Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]]


def bundle_checkpoint(evidence_dir) -> tuple[str, int, str]:
    """(winner, bundle seed, its checkpoint_sha256) from the v2 search evidence; raise if inconsistent."""
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


def model_inputs(input_kind: str, x: np.ndarray, mean, std) -> torch.Tensor:
    """The model input tensor by train_v2's path (selection_tensors): log-Mel standardized once."""
    if input_kind == "logmel":
        x = apply_bin_stats(x, mean, std)
    return torch.from_numpy(np.ascontiguousarray(x))


def score_clips(model, x: torch.Tensor, batch_size: int) -> np.ndarray:
    """Raw scores (float64 logits) on CPU through train_v2's scoring function."""
    return _score(model, x, batch_size, torch.device(DEVICE))


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for a in arrays:
        digest.update(f"{a.dtype.str}{a.shape}".encode())
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


def _waveform_val_b(source: ValBWaveformSource, manifest):
    """Call the aasist val-B source and check it; return (waveforms, labels, paths, subgroups, record)."""
    waveforms, labels, paths = (np.asarray(a) for a in source())
    n = labels.shape[0] if labels.ndim == 1 else -1
    if n <= 0 or labels.dtype != np.int8 or not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{FIT_SPLIT} waveform source: labels must be a non-empty int8 vector of "
                         f"0/1, got {labels.dtype} {labels.shape}")
    if waveforms.dtype != np.float32 or waveforms.shape != (n, WAVEFORM_SAMPLES):
        raise ValueError(f"{FIT_SPLIT} waveform source: waveforms must be float32 "
                         f"{(n, WAVEFORM_SAMPLES)}, got {waveforms.dtype} {waveforms.shape}")
    if not np.isfinite(waveforms).all():
        raise ValueError(f"{FIT_SPLIT} waveform source: waveforms contain NaN or infinite values")
    if paths.dtype.kind != "U" or paths.shape != (n,) or (n > 1 and not (paths[:-1] < paths[1:]).all()):
        raise ValueError(f"{FIT_SPLIT} waveform source: paths must be unicode ({n},), sorted and unique")
    subgroups = _split_rows(manifest, FIT_SPLIT, paths, labels)  # exactly the F2 val-B rows
    record = {"form": "waveform", "n_rows": int(n),
              "content_sha256": _sha256_arrays(waveforms, labels, paths)}
    return waveforms, labels, paths, subgroups, record


def run(checkpoint_dir, selection_set_dir, profile_path, v2_config_path, manifest_path, out_dir, *,
        version: str, cache_dir=None, val_b_waveforms: ValBWaveformSource | None = None
        ) -> dict[str, Any]:
    """Verify the v2 bundle checkpoint (evidence, provenance, M), fit the policy on val-B; return run_report.

    cache_dir: the folder holding the Stage 8 val-B log-Mel cache (val-B/), for cnn-aug and crnn.
    val_b_waveforms: the F3 waveform val-B source, for aasist only. out_dir must not exist; on any
    failure it is removed.
    """
    commit = code_version()["git_commit"]  # before reading inputs or creating outputs
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError(f"output folder already exists: {out_dir}")

    evidence = Path(V2_SEARCH_EVIDENCE)
    winner, bundle_seed, expected_sha256 = bundle_checkpoint(evidence)
    input_kind = INPUTS[winner]
    if input_kind == "waveform":
        if val_b_waveforms is None:
            raise ValueError(f"{winner} takes the F3 waveform: an F3 waveform val-B source is "
                             f"required and none exists yet (it is built only if {winner} wins, "
                             f"Execution Plan v3.0 §9 V4)")
        if cache_dir is not None:
            raise ValueError(f"{winner} takes the F3 waveform: cache_dir (the val-B log-Mel cache) "
                             f"must not be given")
    else:
        if cache_dir is None:
            raise ValueError(f"{winner} takes the log-Mel: cache_dir (Stage 8 val-B log-Mel cache) "
                             f"is required")
        if val_b_waveforms is not None:
            raise ValueError(f"{winner} takes the log-Mel: a waveform val-B source must not be given")

    model, mean, std, ckpt = load_v2_checkpoint(checkpoint_dir, expected_sha256=expected_sha256)
    if ckpt["architecture"] != winner:  # D45: the selected candidate's bundle seed only
        raise ValueError(f"checkpoint architecture {ckpt['architecture']!r} is not the v2 search "
                         f"winner {winner!r}")
    if ckpt["seed"] != bundle_seed:
        raise ValueError(f"checkpoint seed {ckpt['seed']} is not the v2 search bundle seed "
                         f"{bundle_seed}")

    profile_path, v2_config_path = Path(profile_path), Path(v2_config_path)
    manifest_path = Path(manifest_path)
    profile = load_profile(profile_path)
    batch_size = load_v2_config(v2_config_path).require_batch_sizes()[winner]
    manifest = load_f2_manifest(manifest_path)
    profile_config_sha256 = sha256_file(profile_path)
    f2_manifest_sha256 = sha256_file(manifest_path)

    # Provenance checks: the inputs must be the ones the checkpoint was trained and selected with.
    provenance_checks = {}

    def check(key: str, actual: str) -> None:
        provenance_checks[key] = {"expected": ckpt[key], "actual": actual}
        if actual != ckpt[key]:
            raise ValueError(f"provenance check failed: {key} is {actual}, checkpoint records {ckpt[key]}")

    check("profile_fingerprint", profile.fingerprint())
    check("f2_manifest_sha256", f2_manifest_sha256)

    # Reproduction check: weights, statistics and scoring path must give the recorded M exactly.
    selection = load_selection_set(selection_set_dir, profile, form=input_kind, f2_manifest=manifest)
    check("selection_set_sha256", selection.sha256)
    tensors = selection_tensors(selection, input_kind, mean, std)
    eers, _ = evaluate_selection(model, tensors, selection.labels, batch_size, torch.device(DEVICE))
    m = selection_m(eers)
    del tensors, selection
    if m != ckpt["selection_m"]:
        raise ValueError(f"S reproduction failed: recomputed M {m!r}, checkpoint records "
                         f"selection_m {ckpt['selection_m']!r}; recomputed EERs "
                         f"{ {c: eers[c] for c in CONDITIONS} }")

    if input_kind == "logmel":
        features, labels, paths, subgroups, meta = _load_split(
            cache_dir, FIT_SPLIT, profile, profile_config_sha256, manifest)
        val_b_input = {"form": "logmel", "npz_sha256": meta["npz_sha256"], "n_rows": meta["n_rows"]}
    else:
        features, labels, paths, subgroups, val_b_input = _waveform_val_b(val_b_waveforms, manifest)
    x = model_inputs(input_kind, features, mean, std)
    del features
    scores = score_clips(model, x, batch_size)
    if not np.array_equal(scores, score_clips(model, x, batch_size)):
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
        # write_bytes, not write_text: no newline translation, so the bytes (and SHA-256) are the
        # same on every OS (write_text writes CRLF on Windows).
        policy_path.write_bytes(policy_text.encode("utf-8"))
        scores_path = out_dir / VAL_B_SCORES
        _write_npz(scores_path, {"paths": paths, "labels": labels, "score": scores, "p": p})
        run_report = {
            "schema_version": SCHEMA_VERSION,
            "version": version,
            "code_commit": commit,
            "environment": environment(),
            "profile_fingerprint": profile.fingerprint(),
            "profile_config_sha256": profile_config_sha256,
            "f2_manifest_sha256": f2_manifest_sha256,
            "provenance_checks": provenance_checks,
            "v2_search_evidence": {
                name: sha256_file(evidence / name)
                for name in (SEARCH_RUN_REPORT, SEARCH_SELECTION_REPORT)
            },
            "architecture": winner,
            "bundle_seed": bundle_seed,
            "checkpoint_sha256": expected_sha256,
            "val_b_input": val_b_input,
            "scoring": {"device": DEVICE, "batch_size": batch_size},
            "selection_m_recomputed": m,
            "selection_eers_recomputed": {c: eers[c] for c in CONDITIONS},
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
        (out_dir / DIAGNOSTICS_REPORT).write_bytes(_json_text(diagnostics_report).encode("utf-8"))
        (out_dir / RUN_REPORT).write_bytes(_json_text(run_report).encode("utf-8"))
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return run_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path, help="Bundle-seed v2 checkpoint folder")
    parser.add_argument("--selection-set", required=True, type=Path, help="Selection set S folder")
    parser.add_argument("--profile", required=True, type=Path, help="F3 preprocessing profile YAML")
    parser.add_argument("--config", required=True, type=Path,
                        help="configs/v2_candidates.yaml (CPU batch size)")
    parser.add_argument("--manifest", required=True, type=Path, help="Approved F2 FoR manifest")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="cnn-aug, crnn only (required): folder holding the Stage 8 val-B log-Mel cache")
    parser.add_argument("--root", type=Path, default=None,
                        help="aasist only (required): local extracted FoR root; val-B is read through F3")
    parser.add_argument("--out", required=True, type=Path, help="New output folder")
    parser.add_argument("--version", required=True, help="Risk policy version (expected risk-v2)")
    args = parser.parse_args(argv)
    # The winner decides which input argument is required (run's own rule); read before any check.
    winner = bundle_checkpoint(V2_SEARCH_EVIDENCE)[0]
    if INPUTS[winner] == "waveform":
        if args.root is None:
            parser.error(f"{winner} takes the F3 waveform: --root (the FoR root) is required")
        if args.cache_dir is not None:
            parser.error(f"{winner} takes the F3 waveform: --cache-dir must not be given")
        # Imported here: v2_val_b_source imports stage11_v2_run, which imports this module.
        from src.tuning.v2_val_b_source import val_b_waveform_source

        source = {"val_b_waveforms": val_b_waveform_source(args.root, args.manifest, args.profile)}
    else:
        if args.cache_dir is None:
            parser.error(f"{winner} takes the log-Mel: --cache-dir (Stage 8 val-B log-Mel cache) is required")
        if args.root is not None:
            parser.error(f"{winner} takes the log-Mel: --root must not be given")
        source = {"cache_dir": args.cache_dir}
    report = run(args.checkpoint, args.selection_set, args.profile, args.config, args.manifest,
                 args.out, version=args.version, **source)
    print(_json_text(report), end="")


if __name__ == "__main__":
    main()
