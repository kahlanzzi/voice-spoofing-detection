# Stage 11 run

Stage 11. The one FoR test-split run (Protocol v1.0 §7; D31, D41, D45, D46): it scores the test
split with the four models of the D41 model set, stores per-clip scores ([clip_scores.md](clip_scores.md))
and records the run in the registry ([evaluation_harness.md](evaluation_harness.md) "Run registry").
It also holds the D46 dry run on val-A and val-B. This note covers the code and its synthetic
tests; the real run on Colab (notebook 06) and the metrics from the stored files are later PRs. No
FoR test access before D46 is approved.

## Inputs

`src/evaluation/stage11_run.py` holds the official run in two phases,
`claim(*, registry_path, out_dir, results_ref, **inputs)` and
`score(*, root, registry_path, out_dir, run_id, **inputs)`, the dry run
`rehearse(*, root, stage10_scores, out_dir, **inputs)`, and a CLI with the subcommands `rehearse`,
`claim` and `score` (there is no one-step run). `inputs` are the keyword arguments of
`preflight`, shared by all three:

```
python -m src.evaluation.stage11_run rehearse --root <FoR root> \
  --checkpoints <folder with seed_0/ seed_1/ seed_2/> --baseline <baseline-v1 artifact folder> \
  --risk-policy <risk_policy.json> --profile configs/preprocessing.yaml --config configs/cnn.yaml \
  --manifest data/manifests/for2sec_frozen_manifest.csv \
  --stage7-evidence docs/audit_evidence/stage7_baseline \
  --stage8-evidence docs/audit_evidence/stage8_cnn \
  --stage10-evidence docs/audit_evidence/stage10_risk \
  --cache-a <build A> --cache-b <build B> --stage10-scores <val_b_scores.npz> --out <new folder>

python -m src.evaluation.stage11_run claim <same arguments, without --root and --stage10-scores> \
  --registry <run_registry.jsonl> --results-ref <relative path of the run report> --out <new folder>
# notebook: flush Drive (e.g. drive.flush_and_unmount()), re-mount, check the registry file there
python -m src.evaluation.stage11_run score <same arguments as claim, plus --root, without
  --results-ref> --registry <run_registry.jsonl> --run-id <run_id printed by claim> --out <claim's folder>
```

There is no split argument: `claim` and `score` read the test split, fixed in code; `rehearse`
reads val-A and val-B only. `claim` takes no `--root`: it reads no audio.

| Input | Used for |
|---|---|
| checkpoints | `load_checkpoint(seed_<s>/, expected_sha256=...)` for seeds 0, 1, 2 (Stage 8 layout) |
| baseline folder | `artifact_sha256` and `load` (`src/models/baseline.py`) |
| `risk_policy.json` | `RiskPolicy.from_json`, after its SHA-256 check |
| profile, config, manifest | F3 profile, `batch_size`, F2 rows (paths, labels, subgroups, groups) |
| Stage 7, 8, 10 evidence | every expected hash and value (below); nothing is typed in the code except the D41 baseline hash |
| `cache_a`, `cache_b` | two builds of `log_mel/<split>/` and `mfcc/<split>/` from the cache CLIs, same session; `cache_a` also holds val-B for the rehearsal |
| registry (claim, score) | `RunRegistry`; the live file is on Drive (D46 (6)) |
| `run_id` (score only) | the value `claim` printed; 32 lowercase hex, checked as an argument before anything else |
| `results_ref` (claim only) | required, stored in `claim.json`; what `complete` records as the place of the run report. A non-empty, safe, canonical relative POSIX path (`src.freezing.manifests.safe_relative_path`, and the value must equal its canonical form): empty, absolute, drive, `..`, `.` and backslash spellings are refused before anything else |
| `val_b_scores.npz` (rehearse only) | Stage 10 per-clip val-B scores, after its SHA-256 check |

### Evidence read

`load_evidence` reads `run_report.json` of each stage (and Stage 8 `selection_report.json` through
`risk_run.bundle_checkpoint`) and refuses any disagreement:

| Value | Source | Cross-check |
|---|---|---|
| `checkpoint_sha256` seeds 0, 1, 2; bundle seed | Stage 8 | run and selection report agree on the bundle seed; Stage 10 `bundle_seed` and `checkpoint_sha256` equal Stage 8's bundle |
| `artifact_sha256`, `val_b_threshold`, `val_a_eer`, `val_b_eer` | Stage 7 | `artifact_sha256` equals the D41 value; `version` is `baseline-v1` |
| val-A log-Mel `npz_sha256` | Stage 8 | `caches.val-A` equals `val_a_cache_npz_sha256` |
| val-B log-Mel `npz_sha256` | Stage 10 | |
| val-A, val-B MFCC `npz_sha256` | Stage 7 | |
| `risk_policy_sha256`, `val_b_scores_sha256`, `scoring.batch_size` | Stage 10 | `scoring.device` is `cpu` |
| `cnn_config_sha256` | Stage 8 | |
| profile fingerprint, profile config hash, F2 manifest hash | Stages 7, 8, 10 | all three equal |

## Order (D46 (5)), in two phases

Each step raises on failure. Nothing is written before the claim.

### Why two phases

On Colab, Drive writes go to a local cache and are synced later; re-reading the registry on the
same mount reads that cache, so it does not prove that the claim reached Drive (D46 (6)). The run is
therefore split into two CLI calls. Between them, the notebook flushes Drive (unmount with flush,
re-mount) and checks that the registry file on Drive holds the claim. That notebook step is out of
scope here (notebook 06, PR C); this code makes it possible and checks the result: `score`
requires the registry, re-read through the new mount, to be byte-identical to the one `claim`
recorded in `claim.json`.

### Phase 1: `claim` (no test audio is read)

1. **Pre-claim checks** (nothing written), in `claim` and `preflight`: `results_ref` is valid; the
   output folder does not exist; `code_version` gives a clean worktree and a full 40-hex commit;
   environment gate (Python 3.13, torch 2.11.0, D46 (2)); the evidence above; profile
   fingerprint and config hash, F2 manifest hash (equal to the evidence and to the approved
   value), `configs/cnn.yaml` hash and `batch_size`; each checkpoint's hash, computed with the
   repository's function (`cnn_checkpoint.checkpoint_sha256`, the same definition
   `load_checkpoint` uses) and equal to the evidence, then `load_checkpoint` against it (which
   checks the bytes it loads again) and its metadata seed equal to its folder's seed; baseline
   `artifact_sha256`, version and threshold; risk-policy SHA-256; the **reproduction gate**
   (D46 (4)): for log-Mel and MFCC, the val-A caches of `cache_a` and `cache_b` (two folders) load
   and verify, have equal `npz_sha256`, and that hash equals Stage 8 (log-Mel) or Stage 7 (MFCC);
   the **test rows** of the manifest (sorted by path), each valid under the clip_scores row rules,
   reusing its validators (`_check_clip_id`: canonical clip ID; `_check_choice`: label in
   {FAKE, REAL}; `_check_text`: non-empty trimmed group), the columns path, label, subgroup,
   group, split and hash present, and the counts per label and subgroup computed; the registry
   loads and holds no record with `("for", "test", "stage11_test")` for any model set. Every
   check is recorded as `expected` / `actual` / `passed`. The D46 (3) environment record is
   collected here too.
2. **Claim**: `RunRegistry.claim("for", "test", MODEL_IDS, commit,
   evaluation_purpose="stage11_test")`; the registry is re-read from the file and its last record
   must be this `run_id`, `started`, same commit and key.
3. **Environment evidence** (its own step, between the claim and any test read): `<out>/` is
   created (it must not exist), then `environment.json` (the environment record, canonical JSON),
   `pip_freeze.txt` and `claim.json`, each with exclusive creation. `claim.json` holds `run_id`,
   `code_commit`, `registry_sha256` (the registry file after the claim), `results_ref`, the run
   key, the pre-claim checks, the SHA-256 of both environment files and the counts. `claim`
   returns it (the CLI prints it, with the `run_id`).

### Phase 2: `score`

4. **Claim match, then marker**: `<out>/claim.json` (a local file written by `claim`, no test
   access) must exist and name this `run_id`, and its `code_commit` must equal the current commit
   (`code_version`); otherwise `score` is refused and writes nothing, so a mistyped `run_id` costs
   nothing. Then `<out>/scoring_started.json` is created with exclusive creation, before
   `preflight` and everything else. A second `score` call for the same folder is refused, whatever
   happened to the first one, even a crash before any audio read: no rerun after the claim without
   a new decision (D46 (8)). Right after the marker, `<out>/environment_score.json` (the
   environment record of the scoring session, canonical JSON, exclusive creation; recorded, not
   asserted) is written, before any audio read.
5. **Checks again**: `preflight` runs again (same checks as step 1); the test rows are validated
   again and their counts must equal `claim.json`'s; the registry must hold exactly one
   `("for", "test", "stage11_test")` record, `started`, with this `run_id`, this commit and the
   model set (so a completed run, another run_id or no claim is refused); `claim.json` must name
   this `run_id` and commit and a valid `results_ref`; the registry's SHA-256 must equal
   `claim.json`'s `registry_sha256`; both environment files must still match their hashes.
6. **Test features in memory** (D46 (7)): per clip one `read_verified_audio` and one
   `extract_features` call give the log-Mel ([1]) and the MFCC ([2]). Pass 1 keeps the arrays
   and hashes them; pass 2 recomputes only the hashes. Both must be equal. F3 gives finite
   features or raises: the log-Mel is floored before the log (`log_epsilon` 1e-6,
   `src/preprocessing/features.py:39`) and both outputs are checked finite
   (`features.py:41-42`, `:59-60`); a silent clip skips the loudness gain
   (`src/preprocessing/waveform.py:94-95`). If F3 does raise on a test clip, that happens after
   the claim and stops the run under D46 (8); it cannot be checked earlier, because finding it
   needs the test audio.
7. **Scoring on CPU**, each model twice, byte-identical (`tobytes`): baseline
   `BaselineModel.score` on the MFCC; each CNN seed by `risk_run.score_clips` (`apply_bin_stats`
   with the checkpoint's mean and std, then `train_cnn._score`, `batch_size` from the config).
   Bundle-seed `p` and prediction from `RiskPolicy.decide`; baseline prediction from
   `BaselineModel.predict` (D29).
   **Prediction checks** (Entry 99), before anything is written: bundle-seed `p` must equal
   `calibration.transform(raw)` byte for byte, and each prediction must equal the band rule
   written out with `np.where` (the convention of `three_way_decisions`, quoted in the code:
   synthetic if `p >= threshold + half_width`, authentic if `p < threshold - half_width`, else
   uncertain), so the check does not call the code it checks; each baseline prediction must equal
   `synthetic if raw >= threshold else authentic`. The tie rule (`raw == threshold` is
   `synthetic`) is the one of `three_way_decisions` at `half_width` 0 (upper bound
   `threshold + 0.0`, `src/risk/decisions.py:100-102`; `BaselineModel.predict`,
   `src/models/baseline.py:153-155`).
8. **Files**: `<out>/clip_scores/` (new) gets `write_clip_scores` for the four models;
   `manifest_sha256` = the F2 hash; `model_artifact_sha256` = the checkpoint or baseline hash.
   Then `run_report.json`, created exclusively (`open(path, "xb")`).
9. **Complete**: `complete(run_id, results_ref=...)` with the `results_ref` of `claim.json`.

After the claim, any failure raises and leaves the started record; there is no retry and no
cleanup, so whatever was written stays for team review (D46 (8)). Because the environment files
and `claim.json` are written before any audio read, a crash at any later point still leaves the
claim and the environment record of the session that made it.

The environment write is an evidence write between two D46 (5) steps (claim, first test read), not
a reordering: no step moves, nothing is read from the test split, and it carries out D46 (8)
"evidence is kept" for the environment that D46 (3) asks to record.

### Roles (D41, D45)

| Model ID | Role | Columns |
|---|---|---|
| `baseline-v1` | `baseline` | raw score, D29 prediction |
| `cnn-v1-seed<bundle seed>` (seed 0 in the evidence) | `cnn_bundle_seed` | raw score, `p`, three-way prediction |
| the other two seeds | `cnn_other_seed` | raw score |

The bundle seed comes from the evidence, not from a constant.

### Feature hash

Defined once, in `_ArrayDigest`. For each feature: SHA-256 of the header line
`{"dtype":"<f4","name":<name>,"shape":[n, ...]}` (sorted keys, compact) plus `"\n"`, then each
clip's float32 little-endian C-order bytes in path order, which equals the stacked array's bytes.
`features_sha256` is the SHA-256 of the canonical JSON `{"arrays": {log_mel, mfcc digests},
"labels": [...], "paths": [...]}`. Both passes and the rehearsal's cache comparison use it.

## Outputs (`claim`, `score`)

| Output | Content |
|---|---|
| `clip_scores/<model_id>.csv`, `.json` | four models, [clip_scores.md](clip_scores.md) format |
| `environment.json` | the D46 (3) environment record (canonical JSON): `pip_freeze_sha256`, versions of numpy, scipy, librosa, soxr, numba, llvmlite, soundfile, scikit-learn, torch; CPU model, flags and AVX-512 flags from `/proc/cpuinfo` or `"unavailable"`; GPU name and driver or `"none"`; `platform.platform()`; Python; `failures`. Written right after the claim |
| `pip_freeze.txt` | full `pip freeze` text. Written right after the claim |
| `claim.json` | written by `claim` (step 3): `schema_version`, `run_id`, `code_commit`, `registry_sha256`, `results_ref`, `dataset`, `split`, `evaluation_purpose`, `model_ids`, `pre_claim_checks`, `environment_files`, `counts` |
| `scoring_started.json` | written by `score` once `claim.json` matches (step 4): the `run_id` and a UTC timestamp |
| `environment_score.json` | written by `score` right after the marker (step 4): the environment record of the scoring session, same format as `environment.json` |
| `run_report.json` | `schema_version` 1, `code_commit`, `run_id`, `dataset`, `split`, `evaluation_purpose`, `model_set` (role and artifact hash per model), `bundle_seed`, `claim_sha256`, `scoring_marker_sha256`, `pre_scoring_checks` (`expected` / `actual` / `passed`, from `score`'s `preflight`), `environment_files` (SHA-256 of `environment.json` and `pip_freeze.txt`, from `claim.json`), `environment_score_sha256`, `environment_same_as_claim` (`true` if the two canonical records are byte-equal; descriptive only, nothing depends on it), `scoring`, `features` (hashes, `n_clips`), `clip_scores` (`csv_sha256`, `sidecar_sha256`, `n_rows` per model), `counts` (per label and subgroup) |

No metrics: the next PR computes them from the stored files only. `environment_record` never
raises: a missing `/proc/cpuinfo`, `nvidia-smi` or `pip` is recorded under `failures`.

## Rehearsal (`rehearse`, the D46 dry run)

Before anything else, the splits are checked: exactly `val-A` and `val-B`; any other value
(`test`, `Test`, `train`, a string instead of a list) is refused. Then the same `preflight`
(including the reproduction gate), without a registry. For each split, the same
`compute_features` and `score_models` as `score`. Known answers, each `expected` / `actual`, each
failing loudly:

| Check | Expected from |
|---|---|
| in-memory log-Mel and MFCC (with paths and labels) equal the verified caches, val-A and val-B | caches in `cache_a`, whose `npz_sha256` must equal the evidence (val-B log-Mel `2cff949d…`, val-B MFCC `04f91e3e…`) |
| val-A EER per CNN seed | each checkpoint's `val_a_eer` |
| val-A baseline EER | Stage 7 `val_a_eer` |
| val-B bundle-seed raw scores and `p`, byte for byte, same paths and labels | Stage 10 `val_b_scores.npz`, after its SHA-256 equals `val_b_scores_sha256` |
| val-B baseline EER | Stage 7 `val_b_eer` |

On val-B only the bundle seed and the baseline are scored (D45). The baseline prediction check
runs on val-B too. Output: a new folder with `rehearsal_report.json` and `pip_freeze.txt`.

The report is written whatever happens after the split and folder checks. Every check, pre-claim
and known answer, is recorded as `expected` / `actual` / `passed` as it runs. On success the report
has `"passed": true`. On any failure it has `"passed": false`, `failed_check` (the failing check's
name, or `null` if the error was not a recorded check, e.g. a dirty worktree) and `error`, with
every check done so far; then the error is raised again. `"passed": true` is written only when
every recorded check passed.

## Why (team reasoning)

- **(a) Claim before the first test read (D31, D46 (5)).** Reading test audio is test access, so
  the started record exists, and has been re-read from the file, before the first
  `read_verified_audio`. A crash after it blocks a rerun (D46 (8)).
- **(b) Everything checkable is checked before the claim.** A failed check before the claim costs
  nothing; after it, it would burn the one run. So hashes, models, the reproduction gate, every
  test row under the rules `write_clip_scores` will apply, and the registry come first; only work
  that needs test audio comes after.
- **(c) Hashes from committed evidence (as `risk_run.py` (a)).** Each expected value has one
  source, the files reviewed in Stages 7, 8 and 10. The D41 baseline hash is the one constant,
  because D41 names it; it must also equal Stage 7's.
- **(d) Twice, byte-identical (D46 (7), as Stage 10 (c)).** Features and scores are recomputed and
  compared as bytes, so the stored numbers are reproducible on this runtime, not only close.
- **(e) Independent prediction check (Entry 99).** The stored predictions come from the code the
  API uses (`RiskPolicy.decide`, `BaselineModel.predict`); a second computation from the stored
  raw score and `p` must agree before anything is written, so a stored prediction always
  follows from the stored score and the frozen threshold (D29, F4 policy).
- **(f) CPU, all four models (D46 (1)).** The policy was fitted on CPU logits (Stage 10
  `scoring.device`).
- **(g) No metrics here.** The run stores raw outputs; metrics come later from the stored files,
  so they can be recomputed and reviewed without a second test run (D31).
- **(h) Rehearsal as the D46 dry run.** Same code path on validation data with known answers:
  caches (features), checkpoints and Stage 7 (scores via EER), Stage 10 (bundle scores and `p`
  byte for byte). A pass shows that this runtime reproduces the pipeline before the one test run.
- **(i) Group as a string (Entry 99).** `group_id` pairs Stage 14 bootstrap groups; a float or NaN
  (another loader than `load_f2_manifest`) would change the IDs, so it is refused.
- **(j) Environment evidence before the first test read (D46 (3), (8)).** The record is only
  useful if it survives the run it describes; Entry 94 could not explain a drift because nothing
  was recorded. Written right after the claim, it exists for any run that touched test audio.
- **(k) A failed rehearsal still leaves a report.** The dry run exists to find problems before the
  test run; a report with every check done and the first failure shows the team what failed,
  instead of only a traceback.
- **(l) Two phases and a marker (D46 (6), (8)).** A claim is durable only once Drive has synced
  it, which the same mount cannot show; the notebook's flush between the phases makes that
  visible, and `score` checks the registry bytes against `claim.json`. The marker makes "no rerun
  after the claim" hold in code even if `score` crashed before reading any audio; checking
  `claim.json` before it means only the right `run_id` can create it. `environment_score.json`
  records the scoring session too, which may be a new runtime after the flush; whether it equals
  the claim's record is reported, not required.
- **(m) Recorded hashes are computed.** The checkpoint hash in the report is computed with the
  repository's function, not copied from the expected value, so the record shows what was found.

## Limitations

- The guard (`tests/governance/test_for_test_access.py`) now allows this one file; it is a text
  guard, not a sandbox (ADR 0007).
- The rehearsal checks the scoring path through the EER and Stage 10's bytes; it cannot check the
  test split itself.
- The environment record is descriptive; only Python and torch are asserted (D46 (2)). The
  `pytest` and requirements parts of the gate belong to the notebook.
- The Drive flush between `claim` and `score` is a notebook step (PR C); the code checks its
  result (registry bytes equal to `claim.json`) but cannot flush.
- A wrong `run_id` (or a commit other than the claim's) is refused from `claim.json` before the
  marker, so it does not block scoring. Once the marker exists, any further `score` call needs a
  new decision (D46 (8)).
- If F3 raises on a test clip, that happens after the claim and stops the run under D46 (8). It
  cannot be checked earlier without test access; F3 refuses rather than returning NaN or infinity.
- Tested on synthetic data only; the real run uses Colab.

## Tests (`tests/evaluation/test_stage11_run.py`)

Synthetic: a 22-row manifest (val-A, val-B, test), fake audio reads and fake features derived from
each path (monkeypatched), real log-Mel and MFCC caches built from those fakes (build A, build B
and one differing build), three untrained 3-block CNN checkpoints saved with `save_checkpoint`
(their `val_a_eer` is the EER of the runner's scoring path), a real baseline artifact, a real
policy file, a Stage 10 `val_b_scores.npz` and synthetic Stage 7, 8 and 10 evidence.

| Property | Test |
|---|---|
| Each pre-claim check: raises with its own message, no audio read, no registry line, no output | `test_pre_claim_failure_reads_no_audio_and_writes_nothing` (19 cases), `test_every_pre_claim_case_names_its_message` |
| Earlier `stage11_test` claim (other model set) | `test_existing_stage11_claim_is_refused_before_any_read` |
| Existing output folder | `test_existing_out_dir_is_refused_first` |
| Float, NaN, empty or untrimmed group; non-canonical path; missing subgroup column: refused before the claim | `test_bad_group_is_refused_before_the_claim`, `test_non_canonical_test_path_is_refused_before_the_claim`, `test_missing_subgroup_column_is_refused_before_the_claim` |
| `claim` reads no audio; `claim.json` contents | `test_claim_reads_no_audio_and_writes_claim_evidence` |
| `claim` refuses an existing claim | `test_claim_refuses_an_existing_claim` |
| (l) `score` without a claim folder; without a registry claim; with another `run_id` or at another commit (refused with no marker, then the right `run_id` scores); called twice after a crash before any read; after completion; registry, an environment file or the test rows changed since the claim | `test_score_without_a_claim_folder_is_refused`, `test_score_without_a_registry_claim_is_refused`, `test_score_with_another_run_id_is_refused_without_a_marker`, `test_score_at_another_commit_is_refused_without_a_marker`, `test_score_called_twice_is_refused_even_after_a_crash_before_any_read`, `test_score_after_completion_is_refused`, `test_score_refuses_a_registry_changed_since_the_claim`, `test_score_refuses_an_environment_file_changed_since_the_claim`, `test_score_refuses_test_rows_changed_since_the_claim` |
| `environment_score.json` before the first read; `environment_same_as_claim` true and false | `test_environment_files_are_written_between_claim_and_first_read`, `test_environment_same_as_claim_flag` |
| `results_ref` missing, empty, absolute, `..` or non-canonical: refused before the claim | `test_missing_results_ref_is_refused`, `test_bad_results_ref_is_refused_before_the_claim` |
| (a) Claim on disk before the first read; test rows only, twice | `test_claim_is_on_disk_before_the_first_audio_read` |
| (j) Order claim → environment files → first read; a crash on the first read leaves the claim and both files | `test_environment_files_are_written_between_claim_and_first_read`, `test_crash_on_first_audio_read_leaves_claim_and_environment_files` |
| (d) Feature pass 2 differs: raise, record stays started, only the claim evidence and the marker | `test_feature_pass_2_differs_raises_after_claim` |
| (d) Scoring pass 2 differs | `test_cnn_scoring_pass_2_differs_raises`, `test_baseline_scoring_pass_2_differs_raises` |
| (e) Injected prediction mismatch | `test_injected_bundle_prediction_mismatch_raises_before_writing`, `test_injected_baseline_prediction_mismatch_raises_before_writing` |
| Tie rule; band; a `decide()` shifted from the band rule is caught | `test_baseline_tie_rule_is_synthetic`, `test_bundle_predictions_use_the_band`, `test_shifted_decide_prediction_is_caught` |
| Files, roles, report keys, feature hashes, (m) checkpoint hash equal to an independent computation, registry started + completed, second claim refused | `test_happy_path_files_report_and_registry` |
| Feature hash definition | `test_array_digest_is_header_plus_c_order_bytes` |
| Committed evidence gives the D46 values | `test_committed_evidence_loads_with_the_d46_values` |
| Environment record never raises | `test_environment_record_never_raises_without_cpuinfo_nvidia_smi_or_pip`, `test_environment_record_reads_cpu_flags_and_gpu` |
| Rehearsal refuses other splits before reading | `test_rehearse_refuses_other_splits_before_reading` |
| Rehearsal known answers; no registry | `test_rehearse_happy_path_never_touches_a_registry` |
| Rehearsal detects cache, EER and Stage 10 mismatches, and writes a failed report | `test_rehearse_detects_mismatch` (6 cases) |
| (k) Failed rehearsal report: MFCC mismatch with expected/actual hashes; pre-claim failure; non-check error | `test_rehearse_mfcc_mismatch_writes_failed_report_then_raises`, `test_rehearse_preflight_failure_writes_failed_report`, `test_rehearse_non_check_error_writes_failed_report` |
| CLI: `rehearse`, `claim`, `score`; no `run`, no split argument | `test_cli_dispatches_without_a_split_argument` |

The MFCC cache npz-bytes-vs-commit test (D46 (4) precondition) is
`tests/features/test_mfcc_cache_npz_bytes.py`.
