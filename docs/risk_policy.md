# Risk policy

The risk policy turns a raw detector score into the output the system reports: one of
authentic / synthetic / uncertain (FR-05), a confidence (FR-06) and a LOW/MEDIUM/HIGH risk level
(FR-07). It is part of the ModelBundle (D2, AGENTS.md hard rule 2): fitted on FoR val-B only, for
the bundle seed, and frozen in F4 before the external run (Protocol v1.0 §6; D9, D12). It lives in
`src/risk/`, the Risk & Confidence service of the shared core (SDD Fig. 1), so the API and the
research notebooks apply the same rule. It is built in two steps, one section each below; Step R2
(threshold, band width, confidence and risk level) follows.

Step R1 is tested on synthetic scores only; no dataset is read.

## Step R1: band rule in the core, and Platt calibration

### Conventions
- Labels: `1` = spoof (positive class), `0` = bonafide (D4, Protocol v1.0 §5).
- Scores: higher score = more likely spoof. Calibrated output `p` = P(spoof).
- Band rule: unchanged from the evaluation harness (Step 3 of
  [evaluation_harness.md](evaluation_harness.md)); it now lives in `src/risk/decisions.py`.
- Calibration: Platt scaling, 2 parameters, fitted on val-B (Protocol v1.0 §6, FR-18).

`src/risk/decisions.py` holds `AUTHENTIC`, `SYNTHETIC`, `UNCERTAIN`, `DECISIONS`,
`three_way_decisions` and the input checks they share with the harness. `src/evaluation/metrics.py`
imports and re-exports them, so existing imports keep working and both names refer to the same
objects.

`src/risk/calibration.py` holds:
- `PlattCalibration(a, b, n_spoof, n_bonafide)`, a frozen dataclass. `transform(scores)` returns
  `p = 1 / (1 + exp(-(a*s + b)))`, computed so that `exp` is only taken of non-positive numbers
  (no overflow for large `|a*s + b|`). The constructor raises `ValueError` if `a` or `b` is not
  finite, if `a <= 0`, or if a count is not an int ≥ 1.
- `fit_platt(labels, scores) -> PlattCalibration`.

### Method
1. Validate the input with the harness rules: labels 0/1, both classes present, finite scores,
   equal lengths, non-empty; otherwise `ValueError`.
2. Smoothed targets (Platt 1999): `t_spoof = (N_spoof + 1) / (N_spoof + 2)` for spoof rows,
   `t_bonafide = 1 / (N_bonafide + 2)` for bonafide rows.
3. Minimise the cross-entropy `Σ log(1 + exp(z)) − t·z`, with `z = a*s + b`, by Newton's method:
   gradient `Σ (p − t)·(s, 1)`, Hessian `Σ p(1 − p)·(s, 1)(s, 1)ᵀ` plus `1e-12` on the diagonal
   (Lin, Lin and Weng 2007).
4. Start from `a = 0`, `b = log((N_spoof + 1) / (N_bonafide + 1))`, so the starting `p` is the
   (smoothed) class prior.
5. Each Newton step is shortened by a backtracking line search (halving, sufficient decrease
   constant 1e-4) until the loss decreases.
6. Stop when the gradient norm is < 1e-10, or when the Newton decrement `λ²/2 = −½·gᵀd` is ≤ 1e-12,
   with `d` the Newton direction (D44). If `d` is not a descent direction (`λ²/2 < −1e-12`),
   `ValueError` ("not a descent direction"). If an accepted line-search step leaves `(a, b)`
   unchanged, `ValueError` ("stalled"). After 100 iterations without a stop, or if the line search
   fails, `ValueError`.
7. If the fitted `a <= 0`, `ValueError`.

### Why (team reasoning)
- **(a) Core / research layering (SDD Fig. 1).** The API container has the Risk & Confidence service
  but not the Evaluation service, so the band rule cannot live in `src/evaluation`. It moves to
  `src/risk`, `src/risk` never imports `src/evaluation`, and the harness re-exports it. There is
  still one definition of the band rule, and the deployed system does not depend on research code.
- **(b) Smoothed targets.** FoR in-domain may be separable on val-B. Plain logistic regression then
  has no finite solution: `a` grows without limit, every `p` becomes 0 or 1, the confidence is 1.00
  everywhere and the uncertain band collapses. The smoothed targets keep `a` and `b` finite and `p`
  strictly inside (0, 1).
- **(c) Newton in NumPy.** The problem has 2 parameters, so a 2 × 2 Newton solve is exact and fast.
  It needs no new dependency, and the fit is fully specified by this note.
- **(d) `a > 0` enforced.** The calibration is then strictly increasing, so it never changes the
  ranking of scores: ROC-AUC and EER are identical on `s` and on `p`, and score-level results stay
  comparable with the uncalibrated ones. A fit with `a <= 0` would mean the scores rank spoof
  below bonafide on val-B; that is an error to investigate, not something to calibrate away.
- **(e) Stop rule (D44).** The gradient `Σ (p − t)·(s, 1)` carries the units of the score, so an
  absolute bound on its norm means a different accuracy at every score scale; and near the optimum
  of separable data the loss is flat at float64 resolution, steps stop changing `(a, b)` and the
  computed gradient stalls above 1e-10, so a converged fit was reported as "did not converge". The
  Newton decrement `λ²/2` approximates `f − f*` and does not depend on the score scale (Boyd and
  Vandenberghe 2004, §9.5.1). The bound 1e-12 is about 100× above the worst float64 noise floor
  (9.8e-15) measured in the reviewer pre-check on synthetic val-B-shaped logits; a tighter value
  such as 1e-14 has no margin. `λ²/2 = −½·gᵀd` is ≥ 0 only for a descent direction, which a
  positive-definite Hessian guarantees; the descent guard raises when `λ²/2 < −1e-12`, so an uphill
  direction can never pass as "converged". The stall guard turns any remaining case where steps no
  longer move `(a, b)` into an explicit error instead of a silent run to the iteration cap. Limitation: the
  evidence is synthetic only; the real val-B logit shape is unknown until Stage 10.

### Tests (`tests/risk/`)
Each case has its hand reasoning as a comment.

| Point | Test |
|---|---|
| (a) one band rule | `test_metrics_reexports_the_core_band_rule` (`metrics.three_way_decisions is decisions.three_way_decisions`) |
| (a) layering | `test_risk_does_not_import_evaluation` (parses every `src/risk/*.py`; no import of `src.evaluation`) |
| (b) targets | `test_smoothed_targets` (N_spoof = N_bonafide = 2 → 3/4 and 1/4) |
| Method, (c) Newton | `test_parameter_recovery` (5,000 scores ~ N(0, 1), labels ~ Bernoulli(sigmoid(2s − 1)), seed 2026: a = 2 ± 0.15, b = −1 ± 0.15) |
| (b) separable data | `test_separable_data_gives_finite_parameters` (labels [0,0,0,1,1,1], scores [0.1 … 0.9]: finite a > 0, finite b, every p in (0, 1)) |
| (d) ranking kept | `test_calibration_is_monotonic_and_keeps_score_metrics` (200 scores, seed 2026: p strictly increasing; `eer` and `roc_auc` of p equal those of s, tolerance 1e-12) |
| (d) `a > 0` | `test_reversed_scores_raise` |
| Numerical stability | `test_transform_extreme_scores_without_warnings` (scores −1e6, 0, 1e6 with warnings as errors → 0, 0.5, 1) |
| Fail loudly | `test_invalid_input_raises` (one class, NaN score, length mismatch, empty) |
| (e) stop rule | `test_converges_on_val_b_like_separable_logits` (3 val-B-shaped separable kinds × 5 datasets, 660 bonafide / 669 spoof, seed 2026, scales 0.01, 1, 1000: fit succeeds, finite a > 0, ≤ 15 Newton iterations) |
| (e) scale-free | `test_fit_is_invariant_to_score_scale` (one dataset per kind: p for scales 0.01, 1, 1000 agree to 1e-9) |
| (e) at the optimum | `test_fitted_point_is_a_minimum` (one dataset per kind: loss at fitted (a, b) ≤ loss at its 8 neighbours a·(1 ± 1e-4), b ± 1e-4·max(1, \|b\|)) |
| (e) stall guard | `test_stall_guard_turns_spinning_into_a_loud_error` (decrement rule off, every T1 fit: at least one raises "stalled", and every "stalled" message names the Newton decrement) |
| (e) descent guard | `test_non_descent_direction_fails_loudly` (Hessian ridge forced to −1e9: ValueError "not a descent direction") |

## Step R2: fit, apply and save the policy

Step R2 is tested on synthetic scores only; no dataset is read.

### Conventions
- Same labels, scores and `p` as Step R1. The threshold and the band are on calibrated `p`.
- Fit data: val-B only (D2, D9). Maximum deferral on val-B: 10% (Protocol v1.0 §6, D12).
- Confidence: `max(p, 1 − p)` rounded to 2 decimals with Python `round()` (FR-06).
- Risk level: authentic → LOW, uncertain → MEDIUM, synthetic → HIGH (FR-07).

`src/risk/policy.py` (core) holds:
- `RISK_LEVELS` and `SCHEMA_VERSION = 1`.
- `Decision(p, prediction, confidence, risk_level)`, a frozen dataclass.
- `RiskPolicy(version, calibration, threshold, half_width, target_deferral, n_spoof_fit,
  n_bonafide_fit, deferral_on_fit)`, a frozen dataclass. The constructor raises `ValueError` on an
  empty version, a threshold not finite or not strictly inside (0, 1), a negative or non-finite
  `half_width`, `target_deferral` outside [0, 1), a count that is not an int ≥ 1, or
  `deferral_on_fit` outside [0, 1]. It also raises if the policy contradicts itself:
  `n_spoof_fit != calibration.n_spoof`, `n_bonafide_fit != calibration.n_bonafide`, or
  `deferral_on_fit > target_deferral`. The fit never produces these.
- `RiskPolicy.decide(scores)`: one `Decision` per score.
- `RiskPolicy.to_json()` / `RiskPolicy.from_json(text)`: one JSON object with `schema_version` and
  every field (calibration as `{a, b, n_spoof, n_bonafide}`).

`src/tuning/risk_fit.py` (research tier, SDD "Training & Tuning") holds
`fit_risk_policy(labels, scores, *, version, target_deferral=0.10) -> RiskPolicy`.

### Method
1. Calibrate: `calibration = fit_platt(labels, scores)` (Step R1), `p = calibration.transform(scores)`.
2. Threshold: `eer(labels, p)[1]`, the val-B EER point on `p` (D12). If it is not strictly inside
   (0, 1), `ValueError`.
3. Band: `half_width` = the largest value in [0, 1] whose deferral on the fit data is ≤
   `target_deferral`. Deferral is counted with the core `three_way_decisions` and is non-decreasing
   in `half_width`, so bisection on [0, 1] (60 iterations) finds it; `deferral_on_fit` is the rate
   achieved there. Deferral is a step function of `half_width`, so ties or close `p` values can keep
   the achieved rate below the target. `target_deferral = 0` gives `half_width = 0` exactly.
4. Apply (`decide`): `p` from the calibration, the prediction from `three_way_decisions(p,
   threshold, half_width)`, `confidence = round(max(p, 1 − p), 2)`, the risk level from
   `RISK_LEVELS`.
5. Save / load: `to_json` writes floats with `json`'s default `repr`, so every float round-trips
   exactly. `from_json` raises `ValueError` on a wrong `schema_version`, missing or unknown keys
   (top level or calibration), or any value the constructors reject (e.g. NaN threshold).

### Why (team reasoning)
- **(a) Fit in research, apply in core.** The fit uses `eer()` from the harness, so it lives in the
  research tier (`src/tuning`, SDD Training & Tuning). Applying and loading live in `src/risk`, which
  imports neither `src.evaluation` nor `src.tuning`, so the API never imports research code.
- **(b) Threshold on calibrated `p`.** Calibration is strictly increasing (`a > 0`, Step R1 (d)), so
  a threshold on `p` gives the same decisions as the matching threshold on raw scores; the policy
  then works on one scale, `p`, for threshold, band and confidence.
- **(c) Largest band with deferral ≤ 10%.** Protocol v1.0 §6 sets 10% as the maximum deferral on
  val-B, so the band is as wide as that allows and never wider. It is found with the one band rule
  (`three_way_decisions`), not a second implementation, so the fitted deferral is exactly what the
  deployed rule does.
- **(d) Confidence `max(p, 1 − p)`.** Simple and explainable: the calibrated probability of the more
  likely class. **Limitation:** an uncertain case can show a confidence up to about
  `0.5 + half_width`, so the dashboard must show the prediction next to the confidence.
- **(e) Strict JSON.** Exact floats, a schema version and no unknown or missing keys, so the frozen
  bundle (F4) loads exactly what was fitted (NFR-09). The file also cannot contradict itself: the
  fit counts must match the calibration counts and the achieved deferral cannot exceed the target,
  so a hand-edited or mixed-up file fails loudly at load instead of running.

### Tests (`tests/risk/test_policy.py`, `tests/tuning/test_risk_fit.py`, `tests/risk/test_decisions.py`)
Each case has its hand reasoning as a comment.

| Point | Test |
|---|---|
| Risk levels | `test_risk_levels` |
| Method 4, (d) | `test_decide_hand_case` (a = 1, b = 0, threshold 0.5, half_width 0.1: s = 0 → uncertain, MEDIUM, 0.5; s = 3 → synthetic, HIGH, 0.95; s = −3 → authentic, LOW, 0.95) |
| Method 4 | `test_zero_half_width_never_defers` (200 random scores, half_width 0: no uncertain) |
| (e) round trip | `test_json_round_trip_is_exact` (`from_json(to_json(policy)) == policy`) |
| (e) strict load | `test_from_json_rejects_invalid` (wrong schema_version, missing key, unknown key, unknown calibration key, NaN threshold, threshold 0 or 1, negative half_width) |
| Fail loudly | `test_constructor_rejects_invalid`, `test_policy_is_frozen` |
| Method 1–3 | `test_fit_end_to_end_and_deterministic` (665 spoof ~ N(+1, 1), 665 bonafide ~ N(−1, 1), seed 2026: deferral_on_fit in [0.09, 0.10], threshold in (0, 1), refit identical) |
| (c) maximal band | `test_band_is_maximal` (half_width × 1.001 + 1e-12 defers more than the target) |
| (b) threshold | `test_threshold_is_eer_point_and_zero_band_is_forced_binary` (threshold == `eer(labels, p)[1]`; at half_width 0 the decisions match `binary_metrics` counts) |
| Method 3 | `test_zero_target_gives_no_band`, `test_target_outside_range_raises` |
| (a) layering | `test_risk_does_not_import_evaluation`, `test_risk_does_not_import_tuning` |

## References
- [doc] Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to
  regularized likelihood methods.
- [doc] Lin, H.-T., Lin, C.-J. and Weng, R. C. (2007). A note on Platt's probabilistic outputs for
  support vector machines.
- [doc] Boyd, S. and Vandenberghe, L. (2004). Convex Optimization. Cambridge University Press,
  §9.5.1 (the Newton decrement).
