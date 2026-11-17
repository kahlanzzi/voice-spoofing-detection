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
- **Byte-exact `p` re-derivation is platform-bound.** Integrity check 8 compares the recomputed
  `policy.calibration.transform(raw_score)` with the stored `p` byte for byte
  (`rederive.bundle.p_bytes`). On the run's platform (Colab T4, runtime 2026.10, Python 3.13.16,
  Intel Xeon @ 2.00 GHz) the report reproduces byte for byte from the stored files (Entry 161 (b)).
  On another platform the check can stop: on local Windows (Python 3.14.7, numpy 2.5.3) 69 of
  2,658 recomputed `p` differ by at most 1 ulp (max 1.4e-17), with no three-way or forced-binary
  decision change (Entry 161 (a)) [hypothesis: `np.exp` differs across platforms, the same
  processor-bound float effect as D58]. To re-derive the report, run it on the run's platform. The
  check stays byte-exact on purpose (no tolerance was set before the run, and none is added after
  it); the stored `p` values are the ones reported.
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
| Report reads the runner's own synthetic output; D41 check fires on a non-D41 baseline (real `claim` + `score` from `test_stage11_run.py`'s fixtures, imported only with torch) | `test_report_reads_the_runners_own_output` |

## v2 model set

`src/evaluation/stage11_v2_report.py` (Track V, PR V6a) is the metrics report for the separate v2
`stage11_test` claim (D48 (7); Protocol §7 addendum: "the same storage and reporting rules apply"),
computed only from the files `stage11_v2_run score` stored
([stage11_run.md](stage11_run.md#v2-model-set-d48-7)). `stage11_report.py` is unchanged: its output
for the real v1 run is committed and reproduced (Entry 161), and its checks are bound to the v1
model IDs. The v2 module imports the v1 module's model-independent helpers (`CheckFailed`,
`_record` / `_check` / `_fail`, `_sha256`, `_read_json`, `_digest`, `_check_claim`, `_f2_join`, the
metric, CI, formatting and table helpers, `_create`) and writes v2 versions only of what is bound to
v1 IDs, roles or evidence: the run report, registry, score-file, same-clips and re-derivation
checks, the model and seed rows, the Markdown and `write_report` (v1's calls the v1 renderer). One
helper has a v2 copy although it holds no model ID: `_load_manifest` names its check
`manifest.sha256_stage10` and reads the v1 module's approved hash, so the v2 one names it
`manifest.sha256_risk_v2` (same reading rule). No v1 helper was changed. Code and synthetic tests
only; the real report is produced in notebook 09.

**Scope (Donia, option A, 2026-11-17, before any v2 test result):** D57 applied to the v2 model set
as written, with no hypothesis. H1 and H5 are v1 hypotheses (Protocol §1); v2 is compared with v1
in H8 (Stage 13) and H9 (Stage 14). No new decision; the D57 text is unchanged. No baseline row and
no v1–v2 comparison in this report.

### Inputs (v2)

```
build_report(*, run_dir, registry_path, manifest_path, search_evidence_dir,
             risk_evidence_dir) -> dict
write_report(report, out_dir) -> (json_path, md_path)

python -m src.evaluation.stage11_v2_report --run-dir <the folder the v2 score wrote> \
  --registry <run_registry.jsonl> --manifest data/manifests/for2sec_frozen_manifest.csv \
  --search-evidence docs/audit_evidence/v2_search --risk-evidence docs/audit_evidence/v2_risk \
  --out <new folder>
```

| Input | Read for |
|---|---|
| `run_dir/run_report.json`, `run_dir/claim.json` | identity, `architecture`, v2 model set, bundle seed, counts, file hashes |
| `run_dir/clip_scores/<model_id>.csv`, `.json` | the three seeds' per-clip outputs, through `read_clip_scores` |
| registry | the v2 run's started and completed records (the v1 pair and later records are not this run's) |
| F2 manifest | as in v1; bound to risk-v2's `f2_manifest_sha256` and the approved value |
| search evidence | `run_report.json` (`selected`, `bundle_seed`, `checkpoint_sha256`) and `selection_report.json` |
| risk-v2 evidence | `run_report.json` and `risk_policy.json` (the committed file the runner used; no separate argument) |

Torch-free: it does not import `stage11_v2_run` (which imports torch); the runner's constants
(`model_ids_v2`, `model_roles`, candidate names, seeds, file names, `run_dir_name`) are re-declared,
and a torch test checks they equal those of `stage11_v2_run`, `v2_checkpoint`, `v2_search` and
`risk_run`. `search_bundle` restates `v2_risk_run.bundle_checkpoint` without torch (the winner and
bundle seed must agree between the two search reports, and the bundle seed's hash must exist); a
torch test checks both give the same answer or the same refusal in eight cases.

### Integrity checks (v2)

Recorded as in v1 (`expected` / `actual` / `passed`; `CheckFailed` naming the check; nothing
written).

1. **Run report:** v1 check 1 (schema, dataset, purpose, `run_id`, `code_commit`, split string),
   plus `architecture` one of `cnn-aug`, `crnn`, `aasist`; model set = exactly
   `<architecture>-v2-seed0..2` (a v1 or `baseline-v1` ID is refused here); roles one
   `v2_bundle_seed` and two `v2_other_seed`; `bundle_seed` in {0, 1, 2} on the bundle role.
2. **Claim:** v1 check 2 (`stage11_report._check_claim`), plus `architecture`, `bundle_seed` and
   `model_ids` equal to the run report's.
3. **Registry:** the file loads; exactly one started and one completed record for this `run_id`,
   with the run's dataset, split, purpose, the sorted v2 model set and commit; the completed
   `results_ref` equals the claim's. SHA-256 and line count recorded.
4. **Score files:** v1 check 4 for the three seeds; the `clip_scores/` folder holds exactly their
   CSV and sidecar files (a stray `baseline-v1` file is refused).
5. **Same clips** in all three files. 6. **F2 join** (v1's `_f2_join`, reused); the manifest
   SHA-256 equals risk-v2's `f2_manifest_sha256` and the approved F2 value.
7. **Evidence:** `search_bundle` on the search evidence; its winner and bundle seed equal the run
   report's `architecture` and `bundle_seed`; each model's `artifact_sha256` equals the search
   `checkpoint_sha256` of its seed (`<architecture>_seed<k>`); risk-v2 `architecture`, `bundle_seed`
   and `checkpoint_sha256` equal the run's bundle model; risk-v2 `v2_search_evidence` equals the
   SHA-256 of the two search reports; `risk_policy.json` SHA-256 equals risk-v2
   `risk_policy_sha256`; risk-v2 `version` equals the parsed policy's version (`risk_v2.policy_version`,
   the "policy version" binding of `src/bundle/f5.py`).
8. **Re-derivation:** bundle `p` = `calibration.transform(raw_score)` byte for byte; predictions =
   the band rule written with `np.where`; `p` and prediction absent in both `v2_other_seed` files.

### D57 mapped to the v2 model set (`CONVENTIONS_V2`)

| D57 | v2 Stage 11 |
|---|---|
| (1) | "The model" = the bundle-seed v2 model (role `v2_bundle_seed`, seed from the run report); the other two seeds at score level only |
| (2) | EER and ROC-AUC on `raw_score` for all three seeds, never on `p` |
| (3) | Subgroups as D57 (3) |
| (4) | Bundle model: forced binary `binary_metrics(labels, p, risk-v2 threshold)`; three-way `three_way_metrics(labels, stored prediction)`; no baseline row |
| (5)–(6) | Not applicable: no hypothesis is tested; the report says so (`hypotheses: null`, `hypotheses_note`) |
| (7) | CIs as D57 (7) for the bundle model only: overall EER, ROC-AUC, forced-binary FPR, FNR |
| (8) | Seed rows as D57 (8), over the three v2 seeds |
| (9) | As D57 (9) |

### Outputs (v2)

`metrics_report.json` and `metrics_report.md` in a new `--out` folder (exclusive binary writes, clean
worktree, byte-identical on rerun), as v1. JSON keys: v1's minus the hypothesis content, plus
`architecture`, `hypotheses: null` and `hypotheses_note` ("No hypothesis is tested at v2 Stage 11:
H1 and H5 are v1 hypotheses (Protocol §1); v2 is compared with v1 in H8 (Stage 13) and H9 (Stage
14)."). `inputs` holds the SHA-256 of the run report, claim, registry (+ lines), manifest, policy,
risk-v2 run report, both search reports and each CSV / sidecar; `conventions` holds `decision`
"D57", `scope` and the nine mapped items. The Markdown is rendered from the JSON only, with the
note under "Hypotheses" and the line "Reported whatever the outcome (pre-registered)".

### Why (v2)

- **(a) A new module, not an edit of `stage11_report.py`.** The v1 report is committed and
  reproduced byte for byte (Entry 161); an edit would put that reproduction in question. Reuse by
  import (as `stage11_v2_run` does with the v1 runner) keeps one definition of each metric and
  check rule.
- **(b) Option A (no hypothesis).** The Protocol pre-registers H1 and H5 for v1 only and the v1–v2
  comparisons as H8 and H9; applying D57 as written needs no decision round on the critical path to
  F5 (cut-off Fri 4 Dec). Testing H1/H5 on v2, or adding a v1–v2 comparison here, would be a new
  analysis after the Protocol and would need a decision.
- **(c) Evidence binding.** Every number must belong to the search winner's three checkpoints and to
  the policy risk-v2 fitted on that bundle seed; otherwise a valid-looking report could describe
  another candidate or another policy.
- **(d) Torch-free**, as v1.

### Limitations (v2)

- **Byte-exact `p` re-derivation is platform-bound**, as for v1 (above): run the report in notebook
  09 on the run's platform (the runtime that scored the v2 set). Elsewhere `rederive.bundle.p_bytes`
  can stop on 1-ulp differences; no tolerance is added.
- The search and risk-v2 evidence are passed as folders; the real report must point at the committed
  `docs/audit_evidence/v2_search/` and `v2_risk/` (the files the runner read). The binding checks
  refuse other evidence only where it disagrees with the run.
- Few FAKE_other clips and near-zero EERs: as for v1 (above).
- The guard finds no test-split request in the module (it never names the split); `ALLOWED` is
  unchanged. Same blind spot and control as v1.
- Tested on synthetic data only.

### Tests (`tests/evaluation/test_stage11_v2_report.py`)

Synthetic, CPU: a `crnn` v2 set with bundle seed 1 on the v1 hand table (14 clips, 12 groups; the
hand CNN column is the bundle seed, the other two seeds separate perfectly), written with the real
writers, a registry holding the v1 pair first, and search / risk-v2 evidence stubs.

| Property | Test |
|---|---|
| Hand EER, threshold at the EER point, ROC-AUC (overall, per subgroup pair); other seeds score level only | `test_hand_case_score_level` |
| Forced binary per subgroup, three-way confusion, coverage, deferral, `null`; no baseline row | `test_hand_case_decision_level` |
| Bundle-only CIs with counts; seed rows; `hypotheses` null with the note | `test_hand_case_cis_seeds_and_no_hypotheses` |
| Agreement with `eer`, `roc_auc`, `binary_metrics`, `three_way_metrics`, `bootstrap_ci`, `seed_summary` | `test_report_numbers_agree_with_direct_harness_calls` |
| Checks, inputs, registry lines (4), conventions, JSON keys; D57 numbering kept | `test_checks_inputs_and_conventions_are_recorded`, `test_conventions_map_d57_to_v2_with_the_same_numbering` |
| Each integrity check tampered once, raising with its own name, nothing written (62 cases); every recorded check is covered | `test_tampered_input_is_refused_with_its_check`, `test_tampered_run_folder_is_refused_with_its_check`, `test_edited_registry_is_refused`, `test_unapproved_manifest_is_refused`, `test_other_seed_with_p_or_prediction_is_refused`, `test_every_check_name_is_distinct_and_covered` |
| v1 model set, v1 run report, `baseline-v1` entry or file refused | `test_v1_model_set_or_baseline_is_refused`, `test_tampered_input_is_refused_with_its_check[clip_scores.folder_files]` |
| Byte-identical twice (CLI and functions); Markdown = render of the JSON, with the note; collapsed flag | `test_cli_twice_gives_byte_identical_files_rendered_from_the_json`, `test_markdown_flags_collapsed_intervals` |
| Existing `--out`, dirty worktree refused | `test_existing_out_is_refused`, `test_dirty_worktree_is_refused` |
| Torch-free import; v1 helpers imported, not copied; guard finds nothing, `ALLOWED` unchanged; fixture keys = the v2 runner's (`ast`); IDs and roles | `test_import_is_torch_free`, `test_v1_report_module_is_unchanged_and_reused`, `test_report_module_is_not_test_split_access_for_the_guard`, `test_fixture_key_lists_equal_the_v2_runners`, `test_model_ids_and_roles_functions` |
| With torch: constants = the runner's; `search_bundle` = `bundle_checkpoint` (8 cases) | `test_constants_equal_the_v2_runners`, `test_search_bundle_equals_v2_risk_run_bundle_checkpoint` |
| With torch: the report reads the v2 runner's own `claim` + `score` output for `crnn` and `aasist`; another architecture's risk-v2 folder refused | `test_report_reads_the_v2_runners_own_output` |
