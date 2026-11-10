# Stage 11 metrics report

Stage 11, PR B. The metrics of the one FoR test-split run (Protocol v1.0 §7: all D4 metrics with
class counts, per subgroup; H1, Protocol §1; H5, Protocol §7; the seed rows of Protocol §5),
computed **only** from the files the run stored ([stage11_run.md](stage11_run.md)), so they can be
recomputed and reviewed without a second test run (D31; `stage11_run.md` Why (g)). The reporting
details the Protocol does not pin are fixed in **D57** (proposed), before any test result. This note
covers the code and its synthetic tests; running it on the real run is notebook 06 (PR C).

## Inputs

`src/evaluation/stage11_report.py`:

```
build_report(*, run_dir, registry_path, manifest_path, risk_policy_path,
             stage7_dir, stage10_dir) -> dict
write_report(report, out_dir) -> (json_path, md_path)

python -m src.evaluation.stage11_report --run-dir <the folder score wrote> \
  --registry <run_registry.jsonl> --manifest data/manifests/for2sec_frozen_manifest.csv \
  --risk-policy <risk_policy.json> --stage7-evidence docs/audit_evidence/stage7_baseline \
  --stage10-evidence docs/audit_evidence/stage10_risk --out <new folder>
```

| Input | Read for |
|---|---|
| `run_dir/run_report.json`, `run_dir/claim.json` | the run's identity, model set, bundle seed, counts, file hashes |
| `run_dir/clip_scores/<model_id>.csv`, `.json` | the per-clip outputs, through `read_clip_scores` (it verifies each CSV against its sidecar) |
| registry | `RunRegistry(path).records()`: the run's started and completed records |
| F2 manifest | read as `load_f2_manifest` reads it (all columns `str`, canonical paths), without its row-count check so synthetic tests can use it; bound by its SHA-256 instead |
| `risk_policy.json` | `RiskPolicy.from_json`, after its SHA-256 check |
| Stage 7, Stage 10 `run_report.json` | baseline threshold and artifact hash; policy hash, F2 manifest hash, bundle seed and checkpoint hash |

It reads no audio, no feature cache, no checkpoint and no model, and never calls the runner. It is
torch-free: it does not import `src.evaluation.stage11_run` (which imports torch); the constants it
needs (`DATASET`, `PURPOSE`, model IDs, roles, file names, the D41 baseline hash) are re-declared,
and a test checks they equal the runner's. No W&B, no network, no write outside `--out`.

## Integrity checks (before any metric)

Each check is recorded in `report["checks"]` as `expected` / `actual` / `passed` (the runner's
style) and raises `CheckFailed` (a `ValueError`) naming it; nothing is written.

1. **Run report:** `schema_version` 1; `dataset` and `evaluation_purpose` equal the runner's
   constants; `run_id` 32 hex; `code_commit` 40 hex; `split` a non-empty string; model set =
   exactly the four model IDs with roles one `baseline` (on `baseline-v1`), one `cnn_bundle_seed`,
   two `cnn_other_seed`; `bundle_seed` in {0, 1, 2} and its model ID has the bundle role.
2. **Claim:** `claim.json` names the same `run_id`, `code_commit`, `split`, `dataset` and purpose;
   its SHA-256 equals `run_report["claim_sha256"]`; it has a `results_ref`.
3. **Registry:** the file loads (hash chain, replayed rules); exactly one `started` and one
   `completed` record for this `run_id`, each with the run's dataset, split, purpose, sorted model
   set and commit; the completed `results_ref` equals `claim.json`'s. An incomplete run is refused.
   The registry's SHA-256 and line count go into the report (harness "Run registry" (e)).
4. **Score files:** per model, `read_clip_scores`; CSV and sidecar SHA-256 and `n_rows` equal
   `run_report["clip_scores"]`; sidecar `run_id`, purpose, role and artifact hash equal the run
   report's model set; sidecar `manifest_sha256` equals the SHA-256 of `--manifest`.
5. **Same clips:** the same `clip_id`, `label` and `group_id` columns (in the same order) in all
   four files.
6. **F2 join:** every clip is in the manifest with the run's split and the same label (Stage 7's
   `baseline_run._subgroups`, reused) and the same group; the manifest has exactly as many rows of
   that split as the files (no clip missing); counts per label and subgroup equal
   `run_report["counts"]`. The manifest SHA-256 equals Stage 10's `f2_manifest_sha256` and the
   approved F2 value.
7. **Thresholds and evidence:** `risk_policy.json` SHA-256 equals Stage 10's
   `risk_policy_sha256`; Stage 10's `bundle_seed` and `checkpoint_sha256` equal the run's bundle
   model; baseline artifact hash = Stage 7 `artifact_sha256` = the D41 value; baseline threshold =
   Stage 7 `val_b_threshold`.
8. **Re-derivation** (independent of the runner, as its Entry 99 checks): bundle `p` equals
   `policy.calibration.transform(raw_score)` byte for byte; bundle predictions equal the band rule
   written out with `np.where` (synthetic if `p >= t + h`, authentic if `p < t - h`, else
   uncertain); baseline predictions equal `synthetic if raw_score >= threshold else authentic`.

## Metrics (D57 conventions)

| D57 | Convention |
|---|---|
| (1) | "CNN" in H1 and H5 = the bundle-seed CNN (`cnn_bundle_seed`, seed from `run_report.json`); seeds 1–2 at score level only |
| (2) | EER and ROC-AUC on `raw_score` for every model, never on `p` |
| (3) | Subgroup score level: REAL vs FAKE_mp3 and REAL vs FAKE_other. Subgroup decision level: each subgroup alone (REAL: FPR, TN, FP; FAKE_*: recall, FNR, TP, FN) |
| (4) | Bundle CNN: forced binary `binary_metrics(labels, p, policy.threshold)`; three-way `three_way_metrics(labels, stored prediction)`. Baseline: forced binary only, `binary_metrics(labels, raw_score, Stage 7 val_b_threshold)` |
| (5) | H5: forced-binary recall of the bundle CNN; d = recall(FAKE_mp3) − recall(FAKE_other); "clearly lower" iff d ≥ 0.05 and CI lower bound > 0. d ≥ 0.05 is compared exactly on the counts (`fractions.Fraction`) |
| (6) | H1: d = EER(baseline-v1) − EER(bundle CNN), paired; supported iff d > 0 and CI lower bound > 0 |
| (7) | `bootstrap_ci` defaults, `clusters = group_id`, on all rows: bundle CNN and baseline overall EER, ROC-AUC, forced-binary FPR and FNR; H1 d; H5 d. Each with counts, `n_invalid`, `collapsed` (low == high), number of groups. Subgroup rows: point estimates with counts, no CIs |
| (8) | `seed_summary` of test EER and ROC-AUC over seeds 0, 1, 2 (mean, sample SD, values), a separate row |
| (9) | Undefined rates → JSON `null` (Markdown: "undefined"); strict JSON; the EER threshold reported only as "threshold at the EER point (descriptive)"; realised FPR/FNR at the frozen thresholds |

Every statistic is a function of row indices (harness Step 4 (a)); a resample that makes it
undefined (one class, no FAKE_other group drawn) is counted in `n_invalid` by `bootstrap_ci`, never
caught here.

## Outputs

`<out>/metrics_report.json` and `<out>/metrics_report.md`; `<out>` must not exist, both files are
created exclusively (`"xb"`, binary, so the bytes are the same on every OS).

| JSON key | Content |
|---|---|
| `schema_version` | 1 |
| `report_code_commit` | full SHA of the report code (clean worktree required) |
| `run` | `run_id`, run `code_commit`, dataset, split, purpose, bundle seed and model ID, clip and group counts, `counts` |
| `inputs` | SHA-256 of `run_report.json`, `claim.json`, the registry (+ line count), manifest, policy, Stage 7 and Stage 10 reports, and each CSV / sidecar |
| `conventions` | the D57 items as short strings |
| `checks` | every integrity check, `expected` / `actual` / `passed` |
| `models` | per model: role, `score_level` (overall, REAL vs each FAKE subgroup); bundle CNN and baseline: `frozen` thresholds, `forced_binary` (overall, REAL, FAKE_mp3, FAKE_other), `ci`; bundle CNN: `three_way` (overall and per subgroup) |
| `seeds` | EER and ROC-AUC: mean, SD, values per seed |
| `hypotheses` | `H1`, `H5`: rule text, estimate, CI (with counts, `n_invalid`, `collapsed`), conditions, `outcome` "supported" / "not supported" |

The Markdown is rendered from the JSON only (`render_markdown(json.loads(text))`; no number typed
by hand or recomputed): hypothesis, score-level, CI, forced-binary, three-way and seed tables, counts
in every row, "collapsed" next to a collapsed interval, `n_invalid` in its own column, and the line
"Reported whatever the outcome (pre-registered)". Running it twice on the same inputs gives
byte-identical files (no timestamp).

## Why (team reasoning)

- **(a) Stored files only (D31).** The test split is scored once; every number comes from the
  stored scores, so a reviewer can rerun this report and get the same bytes without test access.
- **(b) Check before computing.** A metric from a file that is not the run's (edited CSV, other
  run, other manifest, other policy) would look valid. The checks bind every input to the registry
  record, the run report and the committed Stage 7 / Stage 10 evidence first.
- **(c) Independent re-derivation.** As in the runner (Entry 99): a stored `p` or prediction that
  does not follow from the stored raw score and the frozen threshold is refused, so the decision
  metrics describe the frozen policy, not whatever was written.
- **(d) Conventions as a decision before the result (D57).** Which score, which recall, which
  subsets and which CIs are choices the Protocol leaves open; fixing them in code and in a D-ID now
  means no result can shape them (Protocol §12).
- **(e) Forced-binary recall for H5 (D57 (5)).** The band defers different shares per subgroup, so
  non-deferred recalls would come from denominators the model itself chose; the forced-binary
  recall has the full subgroup as denominator, as H7 uses at its frozen threshold. The non-deferred
  subgroup recall is still reported, in the three-way table.
- **(f) Raw score for score-level metrics (D57 (2)).** Platt is monotonic, so EER and AUC are the
  same in theory, but the sigmoid can saturate to equal floats near 1 and create ties; the raw
  logit is the stored score of record (Stage 10 used it).
- **(g) Exact comparison for "≥ 5 points".** In floats 0.85 − 0.80 = 0.04999…; comparing the counts
  as fractions makes a difference of exactly 5 points count, as the Protocol states.
- **(h) Counts, `n_invalid` and "collapsed" everywhere** (harness Step 4 limitation): near-zero
  EERs give intervals like [0, 0], which do not mean no uncertainty.
- **(i) Torch-free.** The report can run where no model is installed, and importing it cannot load
  one.

## Limitations

- **Guard blind spot.** `tests/governance/test_for_test_access.py` finds no test-split request in
  this module, because it never names the split: it reads it from `run_report.json` and checks it
  against `claim.json` and the registry record. It needs no `ALLOWED` entry, since it reads only the
  stored outputs (D31: later uses read the stored files). The same property means the guard would
  not see a module that read a split name from a file and loaded data with it (a split built at run
  time is the guard's documented blind spot, ADR 0007); here the control is that the module has no
  data loader at all (no audio, cache or model import), and review.
- **Few FAKE_other clips.** About 199 FAKE_other clips are in the test split, so the H5 interval is
  wide (threat model T5); a "not supported" H5 can mean "not shown", not "no MP3 reliance".
- **Near-zero EERs.** With val-A EER 0 on all seeds (Entry 86), test EERs may be near 0 and their
  intervals collapsed; the counts next to them carry the information.
- **Approved F2 value.** The manifest must equal the approved F2 hash (`APPROVED_F2_MANIFEST_SHA256`);
  tests replace it with their synthetic manifest's hash.
- Tested on synthetic data only; the real report is produced by notebook 06 (PR C).

## Tests (`tests/evaluation/test_stage11_report.py`)

Synthetic: a 14-clip hand case (6 REAL, 4 FAKE_mp3, 4 FAKE_other, 12 groups), larger generated
cases (single-clip groups), all written with the real writers (`write_clip_scores`, `RunRegistry`,
`RiskPolicy.to_json`, the runner's JSON format).

| Property | Test |
|---|---|
| Hand-calculated EER, threshold at the EER point, ROC-AUC (overall, per subgroup pair) | `test_hand_case_score_level` |
| Forced binary per subgroup, three-way confusion, coverage, deferral, `null` for undefined rates | `test_hand_case_decision_level` |
| H1 d = 1/8, H5 d = 1/2; "d > 0 but CI includes 0" → not supported; `n_invalid` > 0; seed rows | `test_hand_case_hypotheses_and_seeds` |
| Agreement with `eer`, `roc_auc`, `binary_metrics`, `three_way_metrics`, `bootstrap_ci`, `seed_summary` | `test_report_numbers_agree_with_direct_harness_calls` |
| Checks, input hashes, registry line count, conventions recorded | `test_checks_inputs_and_conventions_are_recorded` |
| Supported H1 and H5; collapsed intervals flagged | `test_supported_case_and_collapsed_intervals` |
| H5 d < 0.05 → not supported; exactly 5 points compared exactly | `test_h5_below_five_points_is_not_supported`, `test_h5_five_points_is_compared_exactly` |
| Each integrity check: one tamper each, raising with its own name, nothing written (39 cases) | `test_tampered_input_is_refused_with_its_check`, `test_tampered_run_folder_is_refused_with_its_check`, `test_edited_registry_is_refused`, `test_unapproved_manifest_is_refused` |
| Incomplete registry run refused | `test_tampered_run_folder_is_refused_with_its_check[registry.records]` |
| Byte-identical output twice (CLI and functions); Markdown = render of the JSON; strict JSON | `test_cli_twice_gives_byte_identical_files_rendered_from_the_json` |
| "collapsed" and `n_invalid` in the Markdown | `test_markdown_flags_collapsed_intervals_and_invalid_resamples` |
| Existing `--out` refused before any work; dirty worktree refused | `test_existing_out_is_refused`, `test_dirty_worktree_is_refused` |
| Torch-free import (fresh interpreter); constants, `_counts` and `_model_set` equal the runner's | `test_import_is_torch_free`, `test_constants_and_formats_equal_the_runners` |
| Fixture writes the runner's `claim.json` / `run_report.json` keys (read from its source with `ast`) | `test_fixture_key_lists_equal_the_runners` |
| Guard finds nothing in this module; `ALLOWED` unchanged | `test_report_module_is_not_test_split_access_for_the_guard` |
