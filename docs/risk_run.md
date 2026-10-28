# Risk-policy run

Stage 10. The official risk-policy fit (Protocol v1.0 §6; D2, D9, D12, D44, D45): it loads the
bundle-seed CNN checkpoint from Stage 8 ([cnn_training.md](cnn_training.md)), checks that it
reproduces its recorded val-A EER, scores FoR val-B and calls `fit_risk_policy`
([risk_policy.md](risk_policy.md)). It writes the policy, the per-clip val-B scores and the reports
that D40 commits in place of W&B logging. This note covers the code and its synthetic tests; the
real run on Colab is a later step.

## Inputs

`src/tuning/risk_run.py` holds `run(cache_dir, checkpoint_dir, profile_path, config_path,
manifest_path, stage8_evidence_dir, out_dir, *, version)` and a CLI:

```
python -m src.tuning.risk_run --cache-dir <folder with val-A/ val-B/> \
  --checkpoint <bundle-seed checkpoint folder> --profile configs/preprocessing.yaml \
  --config configs/cnn.yaml --manifest data/manifests/for2sec_frozen_manifest.csv \
  --stage8-evidence docs/audit_evidence/stage8_cnn --out <new folder> --version risk-v1
```

| Input | Used for |
|---|---|
| `cache_dir/val-A`, `cache_dir/val-B` | log-Mel caches ([cnn_training.md](cnn_training.md), `load_logmel_cache`) |
| checkpoint folder | `load_checkpoint` (safetensors + JSON, non-pickle) |
| `configs/preprocessing.yaml` | F3 profile (fingerprint, `profile_config_sha256`) |
| `configs/cnn.yaml` | `batch_size` only (the training scoring batch size) |
| F2 manifest | paths, labels, completeness and subgroups of both splits |
| `docs/audit_evidence/stage8_cnn/` | `bundle_seed` and `checkpoint_sha256[<bundle seed>]` from `run_report.json`; `bundle_seed` from `selection_report.json` must agree |

## Steps

Each step raises `ValueError` (or the loader's own error) on failure; nothing is written before
step 7.

1. `code_version` refuses a dirty worktree before any input is read; an existing `out_dir` is
   refused.
2. Bundle seed and expected `checkpoint_sha256` are read from the committed Stage 8 evidence (no
   hash in the code). `load_checkpoint(checkpoint_dir, expected_sha256=...)` verifies the bytes it
   loads; the checkpoint metadata `seed` must equal the bundle seed (D45).
3. Provenance checks against the checkpoint metadata: `profile.fingerprint()` must equal
   `profile_fingerprint` and the SHA-256 of the F2 manifest file must equal `f2_manifest_sha256`
   (before any cache is read); after val-A is loaded, its `npz_sha256` must equal
   `val_a_cache_npz_sha256` (before val-B is read). Expected and actual values go to
   `run_report.json` under `provenance_checks`.
   Reproduction check on val-A: the cache is loaded and bound to the profile file
   (`profile_config_sha256`); every cached path must be an F2 val-A row with the same label, and
   every F2 val-A row must be in the cache. The clips are scored on CPU by the training path
   (`apply_bin_stats` with the checkpoint's mean and std → `torch.from_numpy` → `train_cnn._score`,
   `batch_size` from `configs/cnn.yaml`), and `eer()` must equal the checkpoint's `val_a_eer`
   exactly. Only then is val-B loaded.
4. val-B is loaded with the same checks and scored the same way twice; the two float64 arrays must
   be identical.
5. `policy = fit_risk_policy(labels, scores, version=version)`; a refit from the same arrays must
   give a byte-identical `to_json()`.
6. `diagnostics(labels, scores, policy, subgroups)`, a pure function (below).
7. `out_dir` is created and the four files are written; any failure removes it.

## Outputs

| Output | Content |
|---|---|
| `risk_policy.json` | `RiskPolicy.to_json()`, loadable with `RiskPolicy.from_json` |
| `val_b_scores.npz` | `paths` (unicode, sorted), `labels` (int8), `score` (raw logit, float64), `p` (calibrated, float64); deterministic writer, loads with `allow_pickle=False` |
| `diagnostics_report.json` | descriptive only (below) |
| `run_report.json` | `schema_version` 1, `version`, `code_commit`, `environment` (`python`, `numpy`, `torch`), `profile_fingerprint`, `profile_config_sha256`, `f2_manifest_sha256`, `provenance_checks` (`expected` from the checkpoint and `actual` from the inputs, for `profile_fingerprint`, `f2_manifest_sha256` and `val_a_cache_npz_sha256`), `caches` (`npz_sha256`, `n_rows` for val-A and val-B), `bundle_seed`, `checkpoint_sha256`, `n_blocks`, `scoring` (`device` "cpu", `batch_size`), `val_a_eer_recomputed`, `risk_policy_sha256`, `val_b_scores_sha256`, `headline` (`a`, `b`, `threshold`, `half_width`, `deferral_on_fit`, `n_bonafide`, `n_spoof`) |

The two reports have sorted keys and are written with `allow_nan=False`.

### Diagnostics (pre-declared 2026-10-28)

All on val-B, at the fitted policy, with `purpose` "descriptive only":
- `counts`: `n_bonafide`, `n_spoof`.
- `score_level`: `eer` and `roc_auc` on raw scores and on `p`, reported side by side and not
  asserted equal (`p` can tie where the sigmoid saturates); `n_p_equal_0` and `n_p_equal_1` per
  class.
- `separation`: max bonafide and min spoof, raw and `p`; `separable` = max bonafide raw < min
  spoof raw.
- `policy`: `a`, `b`, `threshold`, `half_width`, `target_deferral`, `deferral_on_fit`.
- `decisions`: authentic / uncertain / synthetic counts per class (core `three_way_decisions`);
  `forced_binary`: false positives and false negatives at `half_width` 0.
- `subgroups`: counts, and the raw-score EER of REAL vs FAKE_mp3 and REAL vs FAKE_other, as in
  [baseline_run.md](baseline_run.md).

## Why (team reasoning)

- **(a) Expected hash from committed Stage 8 evidence.** The checkpoint lives on Drive, so it needs
  an external anchor. `checkpoint_sha256` recorded by the Stage 8 run and committed in Git is that
  anchor; reading it from the evidence (not a constant in the code) means the hash has one source,
  and a changed or swapped checkpoint fails before any data is read.
- **(b) val-A reproduction before val-B (verification, not a decision).** The recorded val-A EER is
  re-derived from the stored weights, the stored train-only mean/std and the same scoring code.
  An exact match shows that weights, normalization and scoring path reproduce the selection result,
  so val-B is scored by the model that was selected. Three provenance checks come first and bind
  the inputs to the ones the checkpoint was trained and selected with: the F3 profile fingerprint,
  the F2 manifest SHA-256 and the val-A cache `npz_sha256`, each equal to the value recorded in the
  checkpoint metadata. A matching EER on a different val-A cache, profile or manifest would prove
  nothing; with the checks, the EER is compared on exactly the selection inputs. Nothing is chosen
  or tuned on val-A here; a mismatch only stops the run. The EER depends only on the score ranking, so a CPU rescore of a
  GPU-trained model matches unless the order of clips near the crossing changes; if it fails, the
  run stops and the cause is reported, never relaxed.
- **(c) CPU scoring, twice, identical.** CPU inference is deterministic for fixed inputs and
  threads; scoring twice and comparing proves it for this run, so the scores the policy is fitted
  on are reproducible bit for bit.
- **(d) Refit byte-identical.** The fit (Platt Newton + bisection) is deterministic; refitting from
  the same arrays and comparing `to_json()` checks that the frozen policy is a function of the
  scores alone.
- **(e) Diagnostics pre-declared and descriptive only.** They are listed here before any real
  val-B score exists, and nothing reads them back: no change to the calibration, threshold or band
  may follow from them. Their job is to make visible what the fit did (saturation, separability,
  where the band falls), not to steer it.
- **(f) D45 scope: the bundle seed only.** Only the bundle-seed CNN gets a risk policy; seeds 1
  and 2 are reported at score level in Stage 11. The runner refuses any other seed's checkpoint.
- **(g) Per-clip scores on Drive only.** `val_b_scores.npz` holds one line per val-B clip; it is
  data and stays on Drive (AGENTS.md rule 7). Git holds the reports, whose `val_b_scores_sha256`
  binds them to the file.

## Limitations

- If val-B is separable, the EER threshold on `p` can sit at the spoof edge of the gap (the
  precedent is [results/baseline.md](results/baseline.md) §8, item 1). It is reported in the diagnostics
  (`separation`, `decisions`), never adjusted.
- If the val-A reproduction check or any provenance check fails on the real run, the run stops and
  is reported; proceeding requires a logged decision. The check is never relaxed after the fact.
- Diagnostics are small-sample descriptive values with no confidence interval.
- `code_commit` is the commit of this run; the caches and the checkpoint record their own build
  commits.
- Tested on synthetic data only; the real run uses Colab.

## Tests (`tests/tuning/test_risk_run.py`)

Synthetic data: 37 PCM-16 WAV files (tone/noise mixes per subgroup) over train, val-A, val-B and
one test row that is never read, a manifest whose hashes match, log-Mel caches built with
`build_logmel_cache`, and a 3-block CNN trained 4 epochs on the synthetic train cache with
`train_one`, so the checkpoint's `val_a_eer` is the value training recorded. Seed 2026 for the
data, 0 for training. The F2 manifest loader and approved hash are monkeypatched.

| Property | Test |
|---|---|
| Outputs, keys, hashes, round trip, determinism | `test_end_to_end_outputs_round_trip_and_determinism` |
| val-A then val-B only (D9, D31) | `test_caches_loaded_are_val_a_then_val_b_only` (spy) |
| (a) hash anchor | `test_checkpoint_sha_mismatch_is_refused_before_any_cache_load` |
| (f) D45 | `test_seed_mismatch_is_refused_before_any_cache_load`, `test_inconsistent_stage8_evidence_is_refused` |
| (b) reproduction | `test_val_a_reproduction_mismatch_stops_before_val_b` |
| (b) provenance | `test_provenance_mismatch_stops_before_val_b` (3 cases: `profile_fingerprint` and `f2_manifest_sha256`, no cache loaded; `val_a_cache_npz_sha256`, val-A only; val-B never) |
| F2 completeness | `test_incomplete_val_b_cache_is_refused` |
| Existing folder | `test_existing_out_dir_is_refused` |
| All or nothing | `test_failure_after_mkdir_removes_out_dir` |
| (e) diagnostics | `test_diagnostics_by_hand` (gap, separable flag, saturation, EER/AUC, 3-way counts, forced-binary FP/FN, subgroup EERs) |
| Data separation, no W&B | `test_module_has_no_train_test_external_path_or_wandb` |
| CLI | `test_cli_parses_required_arguments` |
