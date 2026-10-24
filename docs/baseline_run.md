# Baseline run

Stage 7 step 2. The official MFCC/SVM baseline fit (D7, D37): it reads the verified feature caches
of FoR train, val-A and val-B ([feature_cache.md](feature_cache.md)), calls `fit_baseline`
([baseline.md](baseline.md)), saves the non-pickle artifact and writes the reports that D40 commits
in place of W&B logging. The code and the Colab notebook are step 2A; the real run on Colab is a
later step.

## Outputs

`src/tuning/baseline_run.py` holds `run(cache_dir, profile_path, config_path, manifest_path,
out_dir, version)` and a CLI:

```
python -m src.tuning.baseline_run --cache-dir <folder with train/ val-A/ val-B/> \
  --profile configs/preprocessing.yaml --config configs/baseline.yaml \
  --manifest data/manifests/for2sec_frozen_manifest.csv --out <new folder> --version baseline-v1
```

`out_dir` is a new folder holding exactly:

| Output | Content |
|---|---|
| `artifact/` | `baseline.npz` + `baseline.json` (`save`, non-pickle) |
| `selection_report.json` | all 16 grid rows with val-A EER, selected C and gamma, scikit-learn version, SVC settings |
| `threshold_report.json` | D29 threshold, val-B EER there, val-B class counts |
| `run_report.json` | `schema_version` 1, `version`, `code_commit`, `environment` (`python`, `numpy`, `scipy`, `scikit-learn`, `librosa`, `soxr`, `soundfile` versions), `profile_fingerprint`, `profile_config_sha256`, `f2_manifest_sha256`, `baseline_config_sha256`, `caches` (`npz_sha256` and `n_rows` per split), `artifact_sha256`, `selected_C`, `selected_gamma`, `val_a_eer` (at the selected configuration), `val_b_threshold`, `val_b_eer` |
| `subgroup_report.json` | `purpose` (descriptive only), and per val-A / val-B: `n_rows`, `counts` per subgroup, and `eer`, `n_bonafide`, `n_spoof` for REAL vs FAKE_mp3 and REAL vs FAKE_other |

All JSON files have sorted keys and are written with `allow_nan=False`.

## Method

1. `code_version` refuses a dirty worktree before any input is read; an existing `out_dir` is
   refused.
2. The profile (`load_profile`), the config (`load_baseline_config`) and the F2 manifest
   (`load_f2_manifest`, approved hash and schema) are loaded.
3. The three caches are loaded with `load_mfcc_cache` from `cache_dir/train`, `cache_dir/val-A`
   and `cache_dir/val-B`, with the default F2 binding (approved D28 hash) and the given profile
   (fingerprint and version checked). A cache whose recorded `profile_config_sha256` differs from
   the profile file is refused. A missing cache raises.
4. `fit_baseline(train, val-A, val-B, config, version)`: selection on val-A, D29 threshold on
   val-B.
5. Subgroup breakdown: each val-A / val-B cache path is joined to the F2 manifest; a path missing
   from the manifest, or one whose F2 split or label differs from the cache, raises. `eer()` of
   `BaselineModel.score` is computed for REAL vs FAKE_mp3 and REAL vs FAKE_other.
6. Only then is `out_dir` created; the artifact is saved, `artifact_sha256` computed and the four
   reports written. Any failure removes `out_dir`.

## Why (team reasoning)

- **(a) Train, val-A and val-B only (D9, D31).** Selection sees val-A, the threshold val-B. The
  module has no split argument and no test path; the FoR test split is evaluated once at Stage 11
  under the run registry, so nothing here can read it.
- **(b) No W&B (D40).** One deterministic fit; its full provenance is the committed reports, the
  cache JSON files and `artifact_sha256`.
- **(c) Provenance in one file.** `run_report.json` binds the artifact hash to the code commit,
  F3 profile, F2 manifest, baseline config and the exact cache bytes, so a reviewer can trace the
  artifact to every input. It also records the Python and package versions (`environment`):
  feature values depend on librosa/soxr (D39) and the fit on NumPy/SciPy/scikit-learn, and the
  profile fingerprint does not cover library versions, so a rerun can be compared like for like.
- **(d) Subgroup breakdown descriptive and pre-declared.** The per-subgroup EERs show whether the
  baseline behaves differently on MP3-compressed fakes, a known FoR shortcut risk
  ([dataset_audit.md](dataset_audit.md)). They are declared here (2026-10-24) before any real run,
  and nothing reads them back: selection and threshold come only from `fit_baseline`. Declaring
  them first means they cannot be chosen after seeing the results, and they cannot drive a
  decision.
- **(e) All or nothing; deterministic.** A partial output folder could be mistaken for a complete
  run, so nothing is written until everything succeeds. Fixed inputs give byte-identical reports
  and artifact, which the notebook checks by running twice.

## Limitations

- The subgroup EERs on val-A and val-B are small-sample descriptive values with no confidence
  interval.
- `code_commit` is the commit of the run; the caches record their own build commit, which may
  differ if they were built earlier. The notebook builds both from the same commit.
- Tested on synthetic WAV files only; the real fit runs on Colab with
  `notebooks/03_stage7_baseline.ipynb`.

## Colab notebook (`notebooks/03_stage7_baseline.ipynb`)

Same pattern as notebook 02: mount Drive first (drive mode), clone, check out the exact commit
detached, assert a clean worktree, install; verify and extract the pinned FoR archive; run
`pytest -q`; build each cache twice and compare bytes; run the fit twice and compare all files and
`artifact_sha256`; export the evidence to Drive `processed/metadata/stage7/` and the artifact to
`checkpoints/baseline/baseline-v1/` only (never committed to Git).

## Tests (`tests/tuning/test_baseline_run.py`)

Synthetic data: 29 PCM-16 WAV files (tone/noise mixes per subgroup) over train, val-A, val-B and one
test row that is never read, a manifest whose hashes match, and caches built with
`build_mfcc_cache`; seed 2026. The F2 manifest loader and approved hash are monkeypatched to the
synthetic manifest.

| Property | Test |
|---|---|
| Outputs, exact keys, hashes | `test_end_to_end_outputs_keys_and_hashes` |
| Determinism | `test_two_runs_are_byte_identical` |
| Missing cache | `test_missing_cache_is_refused` (each split) |
| Existing folder | `test_existing_out_dir_is_refused` |
| Dirty worktree, nothing read | `test_dirty_worktree_is_refused_before_reading` |
| Profile binding | `test_cache_built_with_another_profile_is_refused` |
| All or nothing | `test_failure_after_fit_removes_out_dir` |
| Manifest join | `test_cache_path_missing_from_manifest_raises` |
| Data separation (D9) | `test_selection_uses_val_a_only_and_threshold_val_b_only` (eer spy; caches loaded: train, val-A, val-B only) |
| Subgroup report | `test_subgroup_report_is_descriptive_and_counts_add_up` |
| CLI | `test_cli_has_no_split_or_test_option` |
