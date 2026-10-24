"""Official MFCC/SVM baseline run (Stage 7 step 2; D7, D9, D29, D31, D37, D40): train, val-A, val-B.

Reads the three verified feature caches (src/features/mfcc_cache.py), fits the baseline with
fit_baseline (selection on val-A, D29 threshold on val-B), saves the non-pickle artifact and writes
four JSON reports into a new folder:
- selection_report.json, threshold_report.json: the fit reports (SelectionReport, ThresholdReport).
- run_report.json: provenance binding the artifact to its inputs (code commit, F3 profile, F2
  manifest, baseline config, cache hashes) and the headline numbers.
- subgroup_report.json: val-A and val-B EER for REAL vs FAKE_mp3 and REAL vs FAKE_other, descriptive
  only (pre-declared); nothing reads it back.

Only train, val-A and val-B exist here: there is no split argument and no test path, and the test
split is evaluated once at Stage 11 (D31). Nothing is logged to W&B (D40); the reports are the
record. Nothing is written unless every step succeeds.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.metrics import eer
from src.features.mfcc_cache import LABEL_CODES, load_mfcc_cache
from src.models.baseline import BaselineModel, artifact_sha256, save
from src.preprocessing.profile import load_profile
from src.preprocessing.validation import (
    F2_MANIFEST_SHA256,
    code_version,
    load_f2_manifest,
    sha256_file,
)
from src.tuning.baseline_fit import fit_baseline, load_baseline_config

SCHEMA_VERSION = 1
RUN_SPLITS = ("train", "val-A", "val-B")
SUBGROUP_SPLITS = ("val-A", "val-B")
FAKE_SUBGROUPS = ("FAKE_mp3", "FAKE_other")
ARTIFACT_DIR = "artifact"
SELECTION_REPORT = "selection_report.json"
THRESHOLD_REPORT = "threshold_report.json"
RUN_REPORT = "run_report.json"
SUBGROUP_REPORT = "subgroup_report.json"
SUBGROUP_PURPOSE = (
    "descriptive only; not used for selection, threshold or any decision (pre-declared 2026-10-24)"
)


def _json_text(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _subgroups(manifest, split: str, paths: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The F2 subgroup of every cached path; raise if a path is missing or disagrees with F2."""
    rows = manifest.set_index("path")
    missing = [p for p in paths.tolist() if p not in rows.index]
    if missing:
        raise ValueError(f"{split}: {len(missing)} cache paths are not in the F2 manifest, "
                         f"first {missing[0]!r}")
    found = rows.loc[paths.tolist()]
    if (found["split"] != split).any():
        raise ValueError(f"{split}: cache paths belong to another F2 split")
    if not np.array_equal(found["label"].map(LABEL_CODES).to_numpy(), labels):
        raise ValueError(f"{split}: cache labels differ from the F2 manifest labels")
    return found["subgroup"].to_numpy()


def _subgroup_split(model: BaselineModel, features, labels, subgroups) -> dict[str, Any]:
    """Counts per subgroup and the EER of REAL vs each FAKE subgroup (one eer() call each)."""
    scores = model.score(features)
    out: dict[str, Any] = {
        "n_rows": int(len(labels)),
        "counts": {name: int((subgroups == name).sum()) for name in ("REAL", *FAKE_SUBGROUPS)},
    }
    for fake in FAKE_SUBGROUPS:
        mask = (subgroups == "REAL") | (subgroups == fake)
        rate, _ = eer(labels[mask], scores[mask])
        out[f"REAL_vs_{fake}"] = {
            "eer": rate,
            "n_bonafide": int((subgroups == "REAL").sum()),
            "n_spoof": int((subgroups == fake).sum()),
        }
    return out


def run(cache_dir, profile_path, config_path, manifest_path, out_dir, version: str) -> dict[str, Any]:
    """Fit the baseline from the three caches and write the artifact and reports; return run_report.

    cache_dir holds train/, val-A/ and val-B/ (one load_mfcc_cache folder each). out_dir must not
    exist; on any failure it is removed.
    """
    commit = code_version()["git_commit"]  # before reading inputs or creating outputs
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError(f"output folder already exists: {out_dir}")
    profile_path, config_path, manifest_path = Path(profile_path), Path(config_path), Path(manifest_path)
    profile = load_profile(profile_path)
    config = load_baseline_config(config_path)
    manifest = load_f2_manifest(manifest_path)
    profile_config_sha256 = sha256_file(profile_path)

    caches = {}
    for split in RUN_SPLITS:
        features, labels, paths, meta = load_mfcc_cache(
            Path(cache_dir) / split, split, profile, expected_f2_manifest_sha256=F2_MANIFEST_SHA256
        )
        if meta["profile_config_sha256"] != profile_config_sha256:
            raise ValueError(
                f"{split}: cache was built from profile config {meta['profile_config_sha256']}, "
                f"profile file is {profile_config_sha256}"
            )
        caches[split] = (features, labels, paths, meta)

    model, selection, threshold = fit_baseline(
        caches["train"][0], caches["train"][1],
        caches["val-A"][0], caches["val-A"][1],
        caches["val-B"][0], caches["val-B"][1],
        config=config, version=version,
    )
    selected = next(r for r in selection.rows
                    if (r.C, r.gamma) == (selection.selected_C, selection.selected_gamma))

    subgroup_report = {"purpose": SUBGROUP_PURPOSE, "splits": {}}
    for split in SUBGROUP_SPLITS:
        features, labels, paths, _ = caches[split]
        subgroups = _subgroups(manifest, split, paths, labels)
        subgroup_report["splits"][split] = _subgroup_split(model, features, labels, subgroups)

    out_dir.mkdir(parents=True)
    try:
        artifact_hash = artifact_sha256(save(model, out_dir / ARTIFACT_DIR))
        run_report = {
            "schema_version": SCHEMA_VERSION,
            "version": version,
            "code_commit": commit,
            "profile_fingerprint": profile.fingerprint(),
            "profile_config_sha256": profile_config_sha256,
            "f2_manifest_sha256": sha256_file(manifest_path),
            "baseline_config_sha256": sha256_file(config_path),
            "caches": {
                split: {"npz_sha256": meta["npz_sha256"], "n_rows": meta["n_rows"]}
                for split, (_, _, _, meta) in caches.items()
            },
            "artifact_sha256": artifact_hash,
            "selected_C": selection.selected_C,
            "selected_gamma": selection.selected_gamma,
            "val_a_eer": selected.val_a_eer,
            "val_b_threshold": threshold.threshold,
            "val_b_eer": threshold.val_b_eer,
        }
        texts = {
            SELECTION_REPORT: selection.to_json(),
            THRESHOLD_REPORT: threshold.to_json(),
            RUN_REPORT: _json_text(run_report),
            SUBGROUP_REPORT: _json_text(subgroup_report),
        }
        for name, text in texts.items():
            (out_dir / name).write_text(text, encoding="utf-8")
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return run_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", required=True, type=Path,
                        help="Folder holding the train, val-A and val-B caches")
    parser.add_argument("--profile", required=True, type=Path, help="F3 preprocessing profile YAML")
    parser.add_argument("--config", required=True, type=Path, help="configs/baseline.yaml")
    parser.add_argument("--manifest", required=True, type=Path, help="Approved F2 FoR manifest")
    parser.add_argument("--out", required=True, type=Path, help="New output folder")
    parser.add_argument("--version", required=True, help="Baseline version, e.g. baseline-v1")
    args = parser.parse_args(argv)
    report = run(args.cache_dir, args.profile, args.config, args.manifest, args.out, args.version)
    print(_json_text(report), end="")


if __name__ == "__main__":
    main()
