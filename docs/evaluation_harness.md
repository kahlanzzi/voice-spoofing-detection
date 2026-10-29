# Evaluation harness

The evaluation harness is three shared modules, `src/evaluation/metrics.py`,
`src/evaluation/bootstrap.py` and `src/evaluation/seeds.py`, used by the MFCC/SVM baseline
(Stage 7), the CNN (Stages 8–11) and the single DEEP-VOICE external run (Stage 13). Because every
result comes from the same code, numbers are comparable across models and datasets (FR-12), and the
positive class is the same everywhere: spoof (D4, Protocol v1.0 §5). The harness was built in five
steps, one section each below.

## Step 1: EER and ROC-AUC

### Conventions
- Labels: `1` = spoof (positive class), `0` = bonafide.
- Scores: higher score = more likely spoof.
- Decision rule: predict spoof if `score >= threshold`.

`roc_auc(labels, scores) -> float` returns the area under the ROC curve, computed by
`sklearn.metrics.roc_auc_score`.

`eer(labels, scores) -> tuple[float, float]` returns `(eer, threshold)`.

### EER method and threshold rule
1. Build the ROC with `sklearn.metrics.roc_curve(..., drop_intermediate=False)` and set
   FNR = 1 − TPR.
2. Find the first ROC index where FPR ≥ FNR. The previous index always exists, because the ROC
   starts at FPR = 0, FNR = 1.
3. If FPR = FNR at that index, that value is the EER and that ROC threshold is the threshold.
4. Otherwise, treat FPR and FNR as straight lines between the previous point and this one. The EER
   is where they cross, at weight `w = gap_before / (gap_before + gap_after)`, where
   `gap_before = FNR − FPR` at the previous point and `gap_after = FPR − FNR` at this point. The
   threshold is interpolated between the two ROC thresholds with the same weight `w`.
5. scikit-learn's first ROC threshold is +∞. It is replaced by the highest score, so the returned
   threshold is always finite.

### Why (team reasoning)
- **(a) scikit-learn for the ROC.** It is widely used and tested, so we do not maintain our own
  ROC code.
- **(b) Interpolation.** On small sets, such as the recordings of one DEEP-VOICE speaker, the ROC has
  few points. Without interpolation, the EER would jump between them.
- **(c) A finite threshold.** Stage 10 stores the threshold in the risk-policy file of the
  ModelBundle, and +∞ cannot be stored or applied meaningfully there.
- **(d) Fail loudly** (AGENTS.md code rules). Both functions raise `ValueError` with a clear
  message on any of these inputs: labels and scores of different lengths, empty input, only one
  class, labels other than 0/1, or NaN/infinite scores.

### Limitation
The EER is an interpolated rate. Applying the returned threshold to the same data gives the error
rates of a neighbouring ROC point, not exactly the EER. For example, in the 0.7 test, a threshold
of 0.7 gives FPR 0 and FNR 0.5, while the EER is 0.5. On large sets (val-B ≈ 1,330 clips) the
gap is small; on small sets (one DEEP-VOICE speaker) it can be large. Stage 11 therefore reports
the realised FPR/FNR at the frozen threshold, and never reports the EER as the error rate at that
threshold.

### Tests (`tests/evaluation/test_metrics.py`)
Each case has its hand calculation as a comment.

| Point | Test |
|---|---|
| Conventions, exact-match rule | `test_perfect_separation` (AUC 1, EER 0, threshold 0.8), `test_reversed_scores` (AUC 0, EER 1), `test_hand_calculated_case` (AUC 0.75, EER 0.5, threshold 0.6) |
| (a) agreement with scikit-learn | `test_roc_auc_matches_sklearn_on_random_scores` (200 scores, seed 2026, tolerance 1e-12) |
| (b) interpolation | `test_all_scores_equal` (EER 0.5 between the two only ROC points), `test_interpolated_threshold_between_roc_points` (EER 0.5, threshold 0.7 between 0.9 and 0.5) |
| Limitation: realised rates | `test_realised_rates_at_interpolated_threshold` (threshold 0.7 gives FPR 0.0, FNR 0.5) |
| (c) finite threshold | `test_all_scores_equal` (the +∞ threshold is replaced; result 0.5, asserted finite) |
| (d) fail loudly | `test_invalid_input_raises` (each invalid input, for both functions) |
| Ranking properties | `test_metrics_invariant_under_monotonic_transform` (logistic transform: calibration such as Platt scaling in Stage 10 is monotonic and must not change score-level metrics), `test_auc_of_negated_scores_is_complement` (AUC(s) + AUC(−s) = 1) |

## Step 2: decision-level metrics at a threshold

### Conventions
- Same labels and scores as Step 1; positive class = spoof (D4, Protocol v1.0 §5).
- Forced binary decision: predict spoof if `score >= threshold` (no uncertain band at this step).

`binary_metrics(labels, scores, threshold) -> BinaryMetrics` returns a frozen dataclass with the
counts `tp, fp, tn, fn, n_spoof, n_bonafide` (int) and the rates `accuracy, precision, recall, f1,
fpr, fnr` (float, fractions in [0, 1]).

### Formulas
| Rate | Formula |
|---|---|
| accuracy | (TP + TN) / N |
| precision | TP / (TP + FP) |
| recall | TP / (TP + FN) |
| F1 | 2TP / (2TP + FP + FN) |
| FPR | FP / (FP + TN) |
| FNR | FN / (FN + TP) |

Any rate whose denominator is 0 is `NaN`, never 0, and no warning is emitted. The threshold must
be a finite number, otherwise `ValueError` is raised. All Step 1 input checks still apply, except
that a single class is allowed.

### Why (team reasoning)
- **(a) Same `>=` rule as the EER threshold.** A threshold returned by `eer()` (and later the
  frozen risk-policy threshold) means the same thing when it is applied here.
- **(b) NaN, not 0, for undefined rates.** A 0 looks like a real result (for example "precision 0"
  when nothing is predicted spoof), and scikit-learn's default silently returns 0 in that case.
  NaN keeps the undefined case visible.
- **(c) One class allowed.** H5 subgroup recall is computed on spoof-only subsets, so
  `binary_metrics` must accept them; rates that need the missing class come out as NaN. EER and
  AUC still need both classes, so `roc_auc` and `eer` keep raising on a single class.
- **(d) F1 from counts.** 2TP / (2TP + FP + FN) is defined in more cases than 2PR / (P + R): for
  example it gives 0 when nothing is predicted spoof but spoofs exist, where precision is NaN.
- **(e) Counts always returned.** Every rate can be recomputed and checked from the counts
  (FR-14).

### Tests (`tests/evaluation/test_metrics.py`)
Each case has its hand calculation as a comment.

| Point | Test |
|---|---|
| Formulas, hand case | `test_binary_metrics_hand_case` (TP = FP = TN = FN = 1; all six rates 0.5) |
| (a) `>=` rule | `test_binary_metrics_tie_counts_as_spoof` (a score equal to the threshold is spoof), `test_binary_metrics_at_eer_threshold` (FPR = FNR = 0.5 at the `eer()` threshold) |
| (b) NaN, (d) F1 from counts | `test_binary_metrics_nothing_predicted_spoof` (precision NaN, F1 0, recall 0, FNR 1) |
| (c) one class | `test_binary_metrics_single_class` (recall 2/3, FPR NaN, `n_bonafide` 0), `test_score_metrics_still_require_both_classes` (`roc_auc`, `eer` still raise) |
| Agreement with scikit-learn, (e) counts | `test_binary_metrics_consistent_with_sklearn` (200 scores, seed 2026: counts sum to N, FNR = 1 − recall, counts match `confusion_matrix`, precision/recall/F1 match `precision_recall_fscore_support`) |
| Fail loudly | `test_binary_metrics_invalid_threshold_raises` (NaN, ±∞, string, None), `test_binary_metrics_invalid_input_raises` (lengths, empty, bad labels, NaN scores) |

## Step 3: three-way decisions and metrics

### Conventions
- Same labels and scores as Steps 1–2; positive class = spoof (D4, Protocol v1.0 §5).
- Decisions use the FR-05 labels, exported as constants: `AUTHENTIC = "authentic"`,
  `SYNTHETIC = "synthetic"`, `UNCERTAIN = "uncertain"`.
- The uncertain band is symmetric around the threshold (D12, Protocol v1.0 §6).

`three_way_decisions(scores, threshold, half_width) -> np.ndarray` returns one decision per score:

| Decision | Rule |
|---|---|
| synthetic | `score >= threshold + half_width` |
| uncertain | `threshold - half_width <= score < threshold + half_width` |
| authentic | `score < threshold - half_width` |

`three_way_metrics(labels, predictions) -> ThreeWayMetrics` returns a frozen dataclass with:
- `confusion`: 2 rows (bonafide, spoof) × 3 columns (authentic, synthetic, uncertain), as counts;
- `n`, `n_deferred` (int); deferred = predicted uncertain;
- `coverage` = (n − n_deferred) / n, `deferral_rate` = n_deferred / n;
- `deferral_rate_bonafide`, `deferral_rate_spoof`: uncertain / class count, per class;
- `non_deferred`: a `BinaryMetrics` over the non-deferred cases only, with synthetic = predicted
  spoof.

Rates are fractions in [0, 1]; any rate whose denominator is 0 is `NaN` and no warning is emitted
(as in Step 2). If every case is deferred, `non_deferred` has zero counts and NaN rates, without an
error. The threshold and `half_width` must be finite numbers and `half_width >= 0`; predictions
must have the labels' length and each be one of the three constants, otherwise `ValueError` is
raised naming the bad value. A single class is allowed, as in Step 2.

### Why (team reasoning)
- **(a) One definition of the band rule.** `three_way_decisions` is the only implementation; the
  Stage 10 risk policy must import it. Evaluation and deployment therefore cannot drift apart.
- **(b) Metrics take predictions, not scores.** The DEEP-VOICE external run stores its per-sample
  outputs (Protocol §8), so they can be evaluated again without re-running the model, which keeps
  the data touched exactly once (AGENTS.md hard rule 1).
- **(c) Boundaries follow the `>=` rule.** A score equal to the upper bound is synthetic, as a
  score equal to the threshold is spoof in Step 2. With `half_width = 0` the band is empty and the
  decisions equal the forced binary decisions of `binary_metrics`.
- **(d) Deferral per class.** A band that mostly defers genuine callers is the T4 concern of the
  threat model (genuine callers treated as suspects). An overall deferral rate could hide it.
- **(e) Both views (D4) from one rate formula.** The forced-binary metrics (`binary_metrics`) and
  the non-deferred metrics both come from `_metrics_from_counts(tp, fp, tn, fn)`, so a rate means
  the same thing in both views.

### Tests (`tests/evaluation/test_metrics.py`)
Each case has its hand calculation as a comment. Tests that produce a NaN rate (including the Step
2 ones) run with `@pytest.mark.filterwarnings("error")`, so "no warning is emitted" is checked in
CI.

| Point | Test |
|---|---|
| (c) boundaries | `test_three_way_decisions_boundaries` (threshold 0.5, half_width 0.1: 0.39 authentic; 0.4, 0.5, 0.59 uncertain; 0.6, 0.7 synthetic) |
| Formulas, hand case | `test_three_way_metrics_hand_case` (confusion (1, 1, 1) / (1, 1, 1); coverage 2/3; deferral 1/3 overall and per class; non-deferred TP = FP = TN = FN = 1, all rates 0.5) |
| (c), (e) `half_width = 0` | `test_zero_half_width_equals_forced_binary` (200 scores, seed 2026: no uncertain, `non_deferred` equals `binary_metrics`) |
| All deferred, NaN | `test_three_way_metrics_all_deferred` (coverage 0, deferral 1, non-deferred counts 0, rates NaN) |
| (d) per-class deferral, one class | `test_three_way_metrics_single_class_deferral_nan` (spoof deferral 0.5, bonafide deferral NaN) |
| Consistency | `test_three_way_metrics_consistency_on_random_scores` (200 scores, seed 2026: confusion sums to n, uncertain column = n_deferred, coverage + deferral = 1, row sums = class counts) |
| Fail loudly | `test_three_way_decisions_invalid_half_width_raises` (negative, NaN, ±∞, string, None), `test_three_way_decisions_invalid_threshold_raises`, `test_three_way_decisions_invalid_scores_raises`, `test_three_way_metrics_invalid_input_raises` (bad prediction strings, non-strings, lengths, empty, bad labels) |

## Step 4: bootstrap confidence intervals

### Conventions
- 95% percentile intervals, 2,000 resamples, seed 2026 (Protocol v1.0 §5, D11); these are the
  defaults of `bootstrap_ci`.
- Resampling by cluster: FoR near-duplicate groups, DEEP-VOICE source speakers (D11).
- Same metric conventions as Steps 1–3; the bootstrap wraps the harness metrics, it does not
  redefine them.

`bootstrap_ci(statistic, n, *, clusters=None, n_resamples=2000, seed=2026, confidence=0.95)
-> BootstrapCI` returns a frozen dataclass with `estimate, low, high, confidence` (float) and
`n_resamples, n_invalid` (int).

### Method
1. `statistic(indices) -> float` receives row indices into the caller's arrays.
2. `estimate = statistic(np.arange(n))`, on the full data. If it raises or is NaN, `ValueError`.
3. Generator: `np.random.default_rng(seed)`, so results are deterministic.
4. Each resample:
   - `clusters=None`: draw `n` row indices with replacement;
   - `clusters` given (length `n`, any hashable ids): draw the unique cluster ids with replacement,
     as many draws as there are clusters, and concatenate all rows of each drawn cluster.
5. A resample is invalid if the statistic raises `ValueError` or returns NaN. It is excluded and
   counted in `n_invalid`. If every resample is invalid, `ValueError`.
6. `low`, `high` = `np.percentile` of the valid statistics at `(1 − confidence)/2` and
   `1 − (1 − confidence)/2`, method `"linear"`.

`ValueError` is also raised for: `n` or `n_resamples` not an int ≥ 1 (bools and floats rejected),
`seed` not an int, `confidence` outside (0, 1), or `clusters` whose length is not `n`.

### Why (team reasoning)
- **(a) Statistic of indices.** One index set per resample feeds both systems of a paired
  difference, so H1, H5 and H6 compare them on the same rows, and a single metric uses the same
  interface.
- **(b) Cluster bootstrap.** Rows inside a cluster (near-duplicates, one speaker's clips) are not
  independent. Resampling rows would treat them as independent and give falsely narrow intervals
  (D11).
- **(c) Protocol values.** Percentile method, 2,000 resamples, seed 2026, 95%, as frozen in
  Protocol v1.0 §5; they are the defaults, so a caller does not repeat them.
- **(d) Estimate on the full data.** The reported number is the metric of the actual test set; the
  resamples only give its uncertainty.
- **(e) Invalid resamples excluded and counted.** A resample with one class has no EER or AUC. It is
  dropped from the percentiles, and `n_invalid` is returned so the drop is visible, never hidden.

### Limitation
With few clusters the percentile interval is approximate and wide. DEEP-VOICE has 8 source
speakers (dataset_audit §8, limitation 4), so its resamples can only recombine 8 clusters. Its
intervals are reported as approximate, with the number of clusters next to them.

Near the boundary (EER or an error rate close to 0 or 1) the percentile interval can collapse,
e.g. to [0, 0], because most resamples reach the same extreme value. This does not mean there is
no uncertainty. Stage 11 therefore reports the error counts (for example FP and FN out of n) next
to every interval, and a collapsed interval is labelled as such. Any n_invalid > 0 is also
reported next to its interval.

### Tests (`tests/evaluation/test_bootstrap.py`)
Each test has its reasoning as a comment.

| Point | Test |
|---|---|
| Determinism | `test_deterministic_for_a_seed` (same seed → identical `BootstrapCI`; seed 7 → different bounds) |
| (d) estimate | `test_estimate_is_statistic_on_full_data` (mean of [1, 2, 3, 4, 10] = 4.0) |
| (c) sanity vs theory | `test_mean_ci_matches_normal_theory` (400 N(0, 1) values, seed 2026: CI contains the mean, width within ±20% of 2·1.96·sd/√n) |
| (b) cluster rule | `test_cluster_resampling_keeps_clusters_whole_and_widens_ci` (20 clusters × 10 identical values: whole clusters only, cluster CI ≥ 2× row CI) |
| (a) paired difference | `test_paired_difference_on_same_resample_is_zero` (low = high = estimate = 0) |
| Harness integration | `test_works_with_harness_eer` (200 random scores, seed 2026: low ≤ EER ≤ high) |
| (e) invalid resamples | `test_invalid_resamples_are_excluded_and_counted` (labels [0, 1], `roc_auc`: 0 < `n_invalid` < 2,000, CI [1, 1]), `test_all_resamples_invalid_raises` |
| Fail loudly | `test_invalid_parameters_raise` (`n_resamples` 0, 10.0, True; confidence 0, 1; clusters of wrong length; float seed; `n` 0, True), `test_nan_estimate_raises` |

## Step 5: seed summary and bundle seed

### Conventions
- CNN seeds 0, 1, 2 on the final configuration (Protocol v1.0 §4, D11).
- Seed reporting: mean ± SD of EER and ROC-AUC across the seeds (Protocol v1.0 §5, D11).
- Bundle seed: the seed with the median val-A EER; tie → lower seed number (Protocol v1.0 §5, D11).

`seed_summary(values) -> SeedSummary` takes a mapping seed → metric value and returns a frozen
dataclass with `mean, sd` (float), `n_seeds` (int) and `values`, the `(seed, value)` pairs sorted
by seed. `select_bundle_seed(val_a_eer) -> int` takes a mapping seed → val-A EER and returns one
seed. Both live in `src/evaluation/seeds.py`.

### Method
1. Input: a mapping of seed → value. Pairs are sorted by seed before use, so the insertion order
   does not matter.
2. `seed_summary`: `mean` = arithmetic mean; `sd` = sample SD (ddof = 1). Needs ≥ 2 seeds.
3. `select_bundle_seed`: `m` = median of the values; among the seeds whose value equals `m`,
   return the lowest seed. Needs an odd number of seeds (≥ 1).

`ValueError` is raised for: an empty mapping or a non-mapping; a seed that is not an int (bools
and floats rejected); a value that is not a finite number (NaN, ±∞, strings, bools rejected);
fewer than 2 seeds in `seed_summary`; an even number of seeds in `select_bundle_seed`.

### Why (team reasoning)
- **(a) Rules as code before any result.** The bundle seed is chosen by a function written and
  tested before any CNN is trained, so the choice cannot be adjusted after seeing the numbers
  (Protocol §12, change control: no change after results are seen).
- **(b) Median, not best.** Picking the seed with the lowest val-A EER would pick the luckiest run;
  the median seed is the typical one, so the bundle is representative of the training recipe.
- **(c) Tie → lower seed.** Two seeds can share the median value; the lower seed number makes the
  choice deterministic.
- **(d) Odd count required.** With an even count the median is the mean of the two middle values
  and may belong to no seed, so there is no "median seed". The function raises instead of guessing.
- **(e) Sample SD, a rough spread.** The 3 seeds are a sample of training runs, so the SD uses
  ddof = 1. With 3 values it is only a rough spread of training randomness and is reported as
  such, as a separate row next to the bootstrap CIs (Protocol §5).

### Tests (`tests/evaluation/test_seeds.py`)
Each case has its hand calculation as a comment.

| Point | Test |
|---|---|
| (e) sample SD | `test_seed_summary_hand_case` ({0: 0.10, 1: 0.12, 2: 0.14}: mean 0.12, sd 0.02) |
| (b) median | `test_select_bundle_seed_median` ({0: 0.30, 1: 0.10, 2: 0.20}: median 0.20 → seed 2) |
| (c) tie | `test_select_bundle_seed_tie_at_median` ({0: 0.2, 1: 0.1, 2: 0.2}: median 0.2 shared by 0 and 2 → 0), `test_select_bundle_seed_all_equal` (all 0.1 → 0) |
| Order independence | `test_order_independence` (reversed insertion order → same seed and summary) |
| (d) odd count | `test_select_bundle_seed_even_count_raises` (2 and 4 seeds) |
| Fail loudly | `test_seed_summary_one_seed_raises`, `test_invalid_input_raises` (empty, NaN, ±∞, string and bool values, bool and float seeds, a list instead of a mapping; both functions) |

## Run registry (D31, key amended by D41)

### Conventions
- Stage 11 runs once (D31): every FoR test-split evaluation is recorded in an append-only run
  registry, and the code refuses a second run for the same model set and purpose.
- A run is identified by `(dataset, split, evaluation_purpose, model_set)` (D41); `model_set` is
  stored sorted.
- `evaluation_purpose` is closed in code: `EVALUATION_PURPOSES = {"stage11_test",
  "stage14_robustness"}` (D41, ADR 0010). Any other value, including `""`, another spelling or
  another case, is refused at `claim` and in a loaded line; adding a purpose needs a new decision.
- The real registry file for Stage 11 is `docs/audit_evidence/run_registry.jsonl`. It is committed
  by PR right after the run. The path is passed by the caller; the code hard-codes none.
- Right after Stage 11, the registry's SHA-256 and line count are recorded in the project-log entry
  and the results PR (mitigation for truncation, see (e)).

`RunRegistry(path)` lives in `src/evaluation/run_registry.py` (research tier; the core `src/risk`
does not import it). `claim(dataset, split, model_set, code_commit, *, evaluation_purpose) ->
run_id`, `complete(run_id, results_ref)` and `records() -> tuple[RunRecord, ...]`.
`evaluation_purpose` is required and keyword-only. `RunRecord` is a frozen
dataclass: `run_id` (uuid4 hex), `status` (`"started"` | `"completed"`), `dataset`, `split`,
`evaluation_purpose` (one of `EVALUATION_PURPOSES`), `model_set` (sorted tuple), `code_commit` (40 lowercase hex), `timestamp` (ISO 8601 UTC),
`results_ref` (`""` while started), `prev_sha256` (64 hex; `"0" * 64` for the first line).

### Method
1. The file is JSON Lines: one record per line, JSON with sorted keys and compact separators,
   UTF-8, terminated by `"\n"`. A missing or empty file is an empty registry.
2. Loading (on construction and on every call) parses each line strictly: unknown or missing keys
   and bad values raise `ValueError` naming the line. It then checks the hash chain: each
   `prev_sha256` must equal the SHA-256 of the previous line's exact bytes, including its `"\n"`.
   A mismatch raises `ValueError` ("registry was edited"). It also replays the rules below, so a
   file the code would never write (a second claim, a completed record without its start) is
   rejected. A line without `evaluation_purpose` (the D31 schema before D41) is a missing key and
   is rejected; no such line exists, since no real registry was written before D41.
3. `claim`: if any record, started or completed, has the same dataset, split, purpose and sorted
   model set, `ValueError` ("already run: a second run needs a new decision"). Otherwise one
   `"started"` record is appended and its `run_id` returned.
4. `complete`: appends one `"completed"` record (same dataset, split, purpose, model set and commit
   as the start, the purpose copied from it) for a run that is started and not yet completed;
   otherwise `ValueError`.
5. Writing opens the file in `"a"` mode (UTF-8, no newline translation) and adds one line; the file
   is never rewritten.

### Why (team reasoning)
- **(a) Claim before run.** The claim is written before the evaluation runs. A run that crashes
  part-way may already have shown test results, so its started record still blocks a rerun; a
  second run needs a new decision.
- **(b) Hash chain.** Each line carries the hash of the previous one, so editing, deleting or
  reordering an earlier line (for example to erase a claim) is detected on the next load.
- **(c) Order-independent model set.** The model set is stored sorted, so listing the same models
  in another order cannot pass as a new run.
- **(d) Path from the caller.** No Colab or Drive path is hard-coded (AGENTS.md code rules); tests
  use a temporary file, and Stage 11 passes the committed evidence path.
- **(e) Limitation.** The code protects one copy of the file: it cannot see a deleted file or a
  second copy. The team-level guarantee is that the real registry is committed through a reviewed
  PR right after Stage 11.
  The hash chain also cannot detect removal of the last line(s): a truncated file is still a valid
  chain. Mitigation (procedure): right after Stage 11, the registry's SHA-256 and line count are
  recorded in the project-log entry and the results PR, so a shortened file no longer matches.
- **(f) Closed purpose in the key (D41).** Stage 14 must score the test split again for the same
  frozen models, so the purpose is part of the key. A free-form purpose would let a caller invent
  a new name and get a new key, bypassing the one-run guard (ADR 0010), so the set is fixed in code
  and unknown values fail closed, at claim and at load.
- **(g) Keyword-only purpose.** `claim` already takes four positional arguments; a required
  keyword-only `evaluation_purpose` cannot be swapped with one of them or forgotten (both are a
  `TypeError` before anything is written), and every call site names the purpose it claims.

### Tests (`tests/evaluation/test_run_registry.py`)
Each test has its reasoning as a comment.

| Point | Test |
|---|---|
| One run per model set, (c) order | `test_second_claim_same_run_raises` (same models; reversed order and another commit also refused) |
| (a) claim before run | `test_crashed_run_still_blocks` (started, never completed → second claim refused) |
| Key = dataset, split, model set | `test_other_model_set_or_split_can_be_claimed` |
| (f) purpose set | `test_purposes_are_exactly_the_d41_set`; `test_unknown_purpose_refused_at_claim` (empty, blank, other name, case variants, padded, non-string; nothing written); `test_malformed_record_raises` (unknown, empty, case-variant, non-string purpose and an old-schema line without it, on load) |
| (f) purpose in the key | `test_same_model_set_under_both_purposes` (two allowed claims); `test_second_claim_same_purpose_refused` (each purpose, first run started or completed); `test_second_claim_same_key_rejected_on_load`; `test_completed_with_other_purpose_rejected_on_load` |
| (g) keyword-only | `test_purpose_is_keyword_only_and_required` (positional or missing → `TypeError`, nothing written) |
| (b) hash chain, purpose | `test_chain_detects_purpose_edit` (an earlier line's purpose changed → "edited") |
| Complete once | `test_complete_once` (second complete, unknown run_id refused; a completed run blocks a claim) |
| Append-only | `test_append_only` (earlier bytes are a prefix after each write; genesis and chain hashes) |
| (b) hash chain | `test_tampering_detected` (edit, delete, reorder an earlier line → "edited") |
| (e) truncation limitation | `test_truncation_is_not_detected_by_the_chain` (last line removed → still loads; file SHA-256 differs from the recorded one) |
| Strict parsing | `test_malformed_record_raises` (unknown/missing key, bad commit, bad status, unsorted model set, non-UTC time, started with a results_ref), `test_malformed_line_raises` (not JSON, not an object, empty line, no final newline), `test_bad_claim_arguments_raise` (nothing written) |
| State in the file | `test_state_lives_in_the_file` (a new `RunRegistry` on the same path sees and enforces earlier claims) |
