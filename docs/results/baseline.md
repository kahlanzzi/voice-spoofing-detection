# MFCC/SVM baseline: validation results

**Validation results (FoR val-A and val-B), not test results; the FoR test split is evaluated once
at Stage 11 (D31).**

Stage 7 step 3. The official fit of the MFCC (mean + std) + RBF-SVM baseline (D7, D37), its D29
threshold and a descriptive subgroup breakdown. Every number below comes from the committed
evidence in [`docs/audit_evidence/stage7_baseline/`](../audit_evidence/stage7_baseline/), cited by
file. Method: [baseline.md](../baseline.md), [baseline_run.md](../baseline_run.md),
[feature_cache.md](../feature_cache.md).

## 1. Provenance

| Item | Value | Source |
|---|---|---|
| Code commit | `297c352c91bfc9daafa2943ba04154194aa33248` (merge of PR #41, Stage 7 step 2A) | `run_report.json`, `determinism.json` |
| Run | Donia, on Colab, 2026-10-24 | |
| Notebook (view-only) | <https://colab.research.google.com/drive/1AFQ4twXQmbW4SXN9iuqU37PZXakaEiGx> | |
| Environment | Python 3.13.16, NumPy 2.5.3, SciPy 1.16.3, scikit-learn 1.9.1, librosa 1.0.0, soxr 1.1.0, soundfile 0.14.0 | `run_report.json` `environment` |
| Version | `baseline-v1` | `run_report.json` |
| Determinism | caches and fit each run twice; byte-identical outputs (`caches_byte_identical: true`, `runs_byte_identical: true`) | `determinism.json` |
| Artifact | `artifact_sha256` `42cc4d53a7cd881f56e4fa54fcf4797de8885b404d0a2b31c293555c868df10a`; stored in Drive `checkpoints/baseline/baseline-v1/`, not in Git | `run_report.json` |

Committed evidence: the three cache JSON files (`train.json`, `val-A.json`, `val-B.json`), the four
reports (`selection_report.json`, `threshold_report.json`, `run_report.json`,
`subgroup_report.json`) and `determinism.json`. The `.npz` caches and the artifact are not
committed (AGENTS.md rule 7); their hashes are.

## 2. Consistency checks [verified]

Checked with a short Python script when the evidence was committed (project log Entry 74).

| Check | Result |
|---|---|
| F2 manifest hash in `run_report.json` and the three cache JSON files = D28 (`badfe948…706f8`) | equal |
| Profile config hash (`7bd35bfd…c8630`) and profile fingerprint (`5b4f5849…d68f2`) in `run_report.json` and the cache JSON files = D39 | equal |
| `baseline_config_sha256` in `run_report.json` = SHA-256 of `configs/baseline.yaml` at this commit | equal |
| Cache row counts (train / val-A / val-B) = Stage 4 split sizes 12,404 / 1,330 / 1,329 (`stage4_split/for2sec_split_report.json`) | equal |
| `npz_sha256` per split in `run_report.json` = `cache_sha256` in `determinism.json` | equal (3/3) |
| SHA-256 of each committed cache JSON = `determinism.json` `cache_sha256` | equal (3/3) |
| SHA-256 of each committed report = `determinism.json` `run_sha256` | equal (4/4) |
| `code_commit` in `determinism.json` = `run_report.json` | equal |

## 3. Model selection (D37)

16 configurations, trained on train (12,404 clips), selected by lowest val-A EER (1,330 clips:
660 bonafide, 670 spoof). Source: `selection_report.json`; class counts from `val-A.json`.

| C | γ | val-A EER | n_support |
|---|---|---|---|
| 0.1 | 0.001 | 18.96% | 8,856 |
| 0.1 | 0.01 | 7.12% | 5,413 |
| 0.1 | 0.025 | 1.67% | 4,057 |
| 0.1 | 0.1 | 0.45% | 4,602 |
| 1 | 0.001 | 13.64% | 5,508 |
| 1 | 0.01 | 1.49% | 2,563 |
| 1 | 0.025 | 0.30% | 1,484 |
| 1 | 0.1 | 0.15% | 2,530 |
| 10 | 0.001 | 7.73% | 3,817 |
| 10 | 0.01 | 0.61% | 1,010 |
| 10 | 0.025 | 0.15% | 771 |
| **10** | **0.1** | **0.15% ← selected** | **2,500** |
| 100 | 0.001 | 2.12% | 2,250 |
| 100 | 0.01 | 0.45% | 564 |
| 100 | 0.025 | 0.15% | 709 |
| 100 | 0.1 | 0.15% | 2,500 |

At two decimals five rows show 0.15%. The exact values differ: (C 1, γ 0.1), (C 10, γ 0.025) and
(C 100, γ 0.025) have 0.0015151515 (= 1/660), while (C 10, γ 0.1) and (C 100, γ 0.1) have
0.0014925373 ≈ 1/670, the lowest. These two tie exactly, and the D37 tie rule (smaller C, then
smaller γ) selects **C 10, γ 0.1** (`selected_C` 10.0, `selected_gamma` 0.1). SVC settings:
RBF kernel, `class_weight` none, `tol` 0.001, scikit-learn 1.9.1 (`selection_report.json`).

## 4. Threshold (D29)

Source: `threshold_report.json`.

| Item | Value |
|---|---|
| Threshold (SVM decision_function, val-B EER point) | 0.13285804202427817 |
| val-B EER | 0.0 |
| val-B clips | 1,329 (660 bonafide, 669 spoof) |

Forced binary only: no calibration and no uncertain band (D29, D7).

## 5. Descriptive subgroup breakdown

Pre-declared on 2026-10-24, before the run; **used for no decision** (`subgroup_report.json`
`purpose`). EER of REAL against each FAKE subgroup separately.

| Split | Comparison | n bonafide | n spoof | EER |
|---|---|---|---|---|
| val-A | REAL vs FAKE_mp3 | 660 | 570 | 0.00% |
| val-A | REAL vs FAKE_other | 660 | 100 | 0.30% |
| val-B | REAL vs FAKE_mp3 | 660 | 569 | 0.00% |
| val-B | REAL vs FAKE_other | 660 | 100 | 0.00% |

The val-A values are optimistic, because the configuration was selected on val-A. val-B was not
used for selection (only for the threshold, which does not change an EER), so it is the fairer
descriptive view.

## 6. Interpretation [hypothesis]

Everything in this section is [hypothesis] unless a source is marked [verified] or [doc].

A val-A EER of about 0.15% and a val-B EER of 0.0 for a 40-value MFCC summary with an RBF-SVM is
"too good" (AGENTS.md: suspect leakage or shortcuts first). It is **not** evidence that synthetic-
speech detection is solved, even in-domain. Three explanations, not mutually exclusive:

1. **In-domain upper bound.** Train, val-A and val-B come from the same FoR-2sec package, so they
   share the same TTS generators and probably the same speakers. FoR has no speaker metadata
   [verified absence, `dataset_audit.md` §8]; the Stage 4 grouping at similarity 0.95 removes only
   near-duplicates, not same-speaker or same-generator clips. In-domain FoR scores are therefore an
   upper bound (`dataset_audit.md` §8, `threat_model.md` T1 and Execution Plan v2.1 §13 (risk:
   leakage between splits) [doc]).
2. **Bandwidth shortcut.** The Stage 1 audit found FAKE clips band-limited: median 99%-energy
   bandwidth about 3.3 kHz (3,264 Hz) against about 5.1 kHz (5,074.5 Hz) for REAL [verified,
   `dataset_audit.md` §3.2]. MFCC means and standard deviations summarise the spectral envelope, so
   they can capture a missing high band. FAKE_other (no MP3 history) separates as well as FAKE_mp3
   (§5 above), so MP3 history alone does not explain the separation. This is H6's concern for the
   CNN.
3. **Loudness and silence cues.** The train-only FAKE-vs-REAL RMS AUC was 0.699 and the
   FAKE_other-vs-REAL silence-fraction AUC about 0.75 [verified, D39]. MFCC coefficient 0 and the
   standard deviations carry level and pause information. This is H7's concern.

**What was not done:** no feature, grid or threshold change after seeing these results (Protocol
§12 change control (3)). Any further check is exploratory and labelled so, or goes through a
pre-registered test.

## 7. Implications

- **H1** (CNN EER < baseline EER on the FoR test, 95% CI of the difference excluding 0; Protocol
  §1) may hit a ceiling: with the baseline near 0% EER in-domain, there is little room for the CNN
  to be measurably better [hypothesis]. H1 is kept unchanged, and its outcome will be reported as
  pre-registered, whatever it is.
- **RQ2 externally.** Under D36 both frozen models are scored on the same external windows in the
  single Stage 13 run. That comparison becomes the main place where RQ2 can show a difference
  between the CNN and the baseline [hypothesis].
- **D41 (not decided here).** Malak is drafting D41: running the baseline through the Stage 14
  degradations, decided before the FoR test split is seen. This document does not decide it.

## 8. Limitations

1. **Weakly determined threshold.** val-B has zero errors at the threshold, so any threshold inside
   the gap between the highest bonafide score and the lowest spoof score separates val-B
   perfectly. The D29 rule was applied correctly, but the data determine the threshold only weakly.
   By the EER method (evaluation_harness.md Step 1, rule 3), when the EER is exactly 0 the
   returned threshold is the ROC threshold at that point, i.e. the lowest val-B spoof score
   [derived from the method]. The threshold therefore sits at the spoof edge of the gap, not in
   its middle: on new data a spoof scoring below every val-B spoof would be missed, while
   bonafide clips have the whole gap as margin, so misses are more likely than false alarms
   [hypothesis].
2. **Grid edge.** The selected γ 0.1 is the largest γ in the D37 grid. val-A EER is flat (about
   0.15%) across several configurations (§3), so the selection is not sensitive to the edge within
   the grid. Nothing outside the grid was tried, and the grid stays as pre-declared (D37).
3. **Uncertainty (descriptive context only).** val-B EER = 0.0 is an exact ROC point
   ([evaluation_harness.md](../evaluation_harness.md) Step 1), so the D29 threshold makes zero
   val-B errors [derived from the EER method; realised counts are not in the reports]. With 0 errors in 660 REAL and 669 spoof clips, the "rule of three" 95% upper
   bounds are about 3/660 ≈ 0.45% for the false-alarm rate and 3/669 ≈ 0.45% for the miss rate.
   These are not the D4 confidence intervals, which are computed on the test split (Protocol §5).
4. **Resolution.** Validation sets of about 1,330 clips cannot distinguish error rates below about
   0.1%: one clip is 1/660 ≈ 0.15% of the bonafide class, and the lowest val-A EER ≈ 1/670 equals
   one clip's share of the spoof class.
