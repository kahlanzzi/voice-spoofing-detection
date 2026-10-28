# Risk-policy fit: validation results (`risk-v1`)

**Validation results (FoR val-B), not test results.** The policy is fitted on val-B only, for the
bundle-seed CNN, and frozen as fitted: the rules are fixed by Protocol v1.0 §6, D12, D44 and D45
[doc]. The FoR test split is evaluated once at Stage 11 (D31). Nothing here touches the external
set.

Stage 10. The official fit of the risk policy (Platt calibration, EER threshold on `p`, uncertain
band at most 10% deferral) on FoR val-B with the Stage 8 bundle-seed checkpoint. Every number below
comes from the committed evidence in
[`docs/audit_evidence/stage10_risk/`](../audit_evidence/stage10_risk/), cited by file. Method:
[risk_run.md](../risk_run.md), [risk_policy.md](../risk_policy.md). Model:
[cnn_selection.md](cnn_selection.md). Baseline counterpart: [baseline.md](baseline.md).

Tags: [verified] read from or checked against the committed files; [doc] stated by a project
document; [derived] computed from committed values (the computation is shown); [inference] a
reading of the numbers; [hypothesis] an untested explanation.

## 1. Provenance

| Item | Value | Source |
|---|---|---|
| Code commit | `11c07fedcd755dc72f16b2695f180cf094aeca04` (merge of PR #59, notebook 05) [verified] | `run_report.json`, `stage10_export_manifest.json` |
| Run | Donia, on Colab (T4 runtime, scoring on CPU), 2026-10-28; manifest written 20:26:08 UTC [verified] | `stage10_export_manifest.json`; runtime: §6 |
| Environment | Python 3.13.16, NumPy 2.5.3, torch 2.11.0+cu130 [verified] | `run_report.json` `environment` |
| Version | `risk-v1` (CNN `cnn-v1`) [verified] | `run_report.json`, manifest |
| Tests in the run | 760, 0 failures / errors / skipped [verified] | manifest `pytest_counts` |
| Data | F2 manifest `badfe948…706f8` (D28); F3 profile `7bd35bfd…c8630`, fingerprint `5b4f5849…d68f2` (D39); FoR revision `ff8c82c7…9055` (D21) [verified] | `run_report.json`, manifest |
| Checkpoint | bundle seed 0, `n_blocks` 4, `checkpoint_sha256` `c33eb820…958932` = Stage 8 seed 0 [verified] | `run_report.json`, `stage8_cnn/run_report.json` |
| Caches | val-A 1,330 rows, `npz_sha256` `4502eea2…862c81`; val-B 1,329 rows, `2cff949d…c568d7`; two builds byte-identical [verified] | `run_report.json` `caches`, manifest `cache_sha256` |
| Scoring | CPU, batch size 256 [verified] | `run_report.json` `scoring` |

### Checks that passed

| Check | Result | Source |
|---|---|---|
| Checkpoint bytes = committed Stage 8 `checkpoint_sha256` (seed 0) | equal [verified] | `run_report.json`, `stage8_cnn/run_report.json` |
| Checkpoint seed = bundle seed 0 (D45) | `bundle_seed` 0 [verified] | `run_report.json` |
| `profile_fingerprint` expected = actual | equal [verified] | `run_report.json` `provenance_checks` |
| `f2_manifest_sha256` expected = actual | equal [verified] | `run_report.json` `provenance_checks` |
| `val_a_cache_npz_sha256` expected = actual (= Stage 8 `4502eea2…`) | equal [verified] | `run_report.json` `provenance_checks` |
| val-A EER recomputed on CPU = checkpoint `val_a_eer` | 0.0 = 0.0 [verified] | `run_report.json` `val_a_eer_recomputed`; [cnn_selection.md](cnn_selection.md) §4 |
| val-B scored twice, identical float64 arrays | passed [doc] | the runner raises otherwise and writes nothing ([risk_run.md](../risk_run.md) steps 4, 7); no field records it |
| Refit from the same arrays, byte-identical `to_json()` | passed [doc] | same (step 5) |
| Both cache builds byte-identical | equal [verified] | manifest `cache_sha256.cache_a` = `cache_b` |
| Exported files after the Drive remount; committed bytes | equal [verified] | [evidence README](../audit_evidence/stage10_risk/README.md), "Verification" |

## 2. The policy

Source: `risk_policy.json` (= `run_report.json` `headline` = `diagnostics_report.json` `policy`)
[verified].

| Item | Value |
|---|---|
| Platt `a` | 1.0403213615577922 |
| Platt `b` | 5.625181819635712 |
| Fit counts | 669 spoof, 660 bonafide |
| Threshold on `p` (val-B EER point, D12) | 0.6333105553100875 |
| `half_width` | 0.3636377690579665 |
| Target deferral | 0.10 |
| Achieved deferral on val-B | 0.09932279909706546 = 132 of 1,329 |

The band and the threshold on the raw-logit scale [derived]. `p = sigmoid(a·s + b)`, so the raw
score for a value of `p` is `s = (logit(p) − b) / a`. Computed with a short Python script from
`risk_policy.json` (sigmoid of each raw value gives back `p` to 1e-16):

| Point | `p` | Raw logit `s` |
|---|---|---|
| Lower band edge (threshold − `half_width`) | 0.269672786252121 | −6.3648266917875995 |
| Threshold | 0.6333105553100875 | −4.881891679664631 |
| Upper band edge (threshold + `half_width`) | 0.996948324368054 | 0.15747661888600895 |

With the `>=` rule of the band ([evaluation_harness.md](../evaluation_harness.md) Step 3, rule table and Why (c)) [doc],
a clip is synthetic if `p` ≥ 0.99695 (raw ≥ 0.157), authentic if `p` < 0.26967 (raw < −6.365), and
uncertain in between [derived]. The achieved deferral, 132 / 1,329 = 0.0993228, equals
`deferral_on_fit` and is below the 10% target because deferral is a step function of `half_width`
([risk_policy.md](../risk_policy.md) Step R2, Method 3) [derived, doc].

## 3. Diagnostics (val-B)

Pre-declared on 2026-10-28, before any val-B score existed; **descriptive only**, used for no
decision (`diagnostics_report.json` `purpose`; [risk_run.md](../risk_run.md) Why (e)). All values
[verified] from `diagnostics_report.json`.

| Group | Item | Value |
|---|---|---|
| Counts | bonafide / spoof | 660 / 669 |
| Score level, raw | EER / ROC-AUC | 0.001494768310911776 / 0.9999841463967024 |
| Score level, `p` | EER / ROC-AUC | 0.001494768310911776 / 0.9999841463967024 |
| Saturation | `p` = 0 (bonafide / spoof) | 0 / 0 |
| Saturation | `p` = 1 (bonafide / spoof) | 0 / 0 |
| Separation | max bonafide, raw / `p` | −4.887398719787598 / 0.6319790848927088 |
| Separation | min spoof, raw / `p` | −7.173023223876953 / 0.1373984873573772 |
| Separation | separable | false |
| Decisions, bonafide | authentic / uncertain / synthetic | 657 / 3 / 0 |
| Decisions, spoof | authentic / uncertain / synthetic | 1 / 129 / 539 |
| Forced binary (`half_width` 0) | false positives / false negatives | 0 / 1 |
| Subgroups | REAL / FAKE_mp3 / FAKE_other | 660 / 569 / 100 |
| Subgroup EER, raw | REAL vs FAKE_mp3 | 0.0 |
| Subgroup EER, raw | REAL vs FAKE_other | 0.010000000000000009 |

### Observations (descriptive)

1. **val-B is not separable, by one clip.** The lowest spoof score (raw −7.17) is below the highest
   bonafide score (raw −4.89) [verified]. Exactly one spoof clip scores below the highest bonafide
   [derived: at least one, the minimum; at most one, because the forced-binary threshold
   (raw −4.882) is above the highest bonafide and has 1 false negative]. That clip is FAKE_other
   [derived: REAL vs FAKE_mp3 has EER 0, so every FAKE_mp3 clip scores above every REAL clip]. The
   overall EER, 0.0014948, is 1/669 ≈ 0.0014948 up to float rounding [derived].
2. **Forced binary: 0 false positives, 1 false negative** [verified]. The threshold lies 0.0055 raw
   above the highest bonafide score (−4.8819 vs −4.8874) [derived].
3. **The band holds mostly spoof clips.** 129 of the 132 deferred clips are spoof: 129 / 669 =
   19.3% of spoof and 3 / 660 = 0.45% of bonafide [derived]. The one spoof clip below the lower band
   edge (raw −7.17 < −6.365) is the one decided authentic [derived]. So on val-B a clip is called
   "synthetic" only when `p` ≥ 0.99695; 539 of 669 spoof clips reach it, and no bonafide clip
   does [derived].
4. **Confidence of uncertain cases.** Confidence is `max(p, 1 − p)` rounded to 2 decimals
   ([risk_policy.md](../risk_policy.md) Step R2, Method 4) [doc]. An uncertain clip just below the
   upper band edge has `p` near 0.997, so its confidence can display as 1.00 [derived]. This is the
   limitation recorded in [risk_policy.md](../risk_policy.md) Why (d): an uncertain case can show a
   confidence up to max(threshold + `half_width`, 1 − (threshold − `half_width`)), rounded, so the
   dashboard must show the prediction next to the confidence [doc]. Here that bound is
   max(0.99695, 0.73033) = 0.997 [derived].
5. **Raw and `p` metrics are identical and nothing saturates.** EER and ROC-AUC are equal on raw
   scores and on `p`, and no clip has `p` exactly 0 or 1 [verified], as expected for a strictly
   increasing calibration with `a` > 0 and smoothed targets ([risk_policy.md](../risk_policy.md)
   Why (b), (d)) [doc].

## 4. Interpretation [inference]

Everything in this section is [inference] unless a source is marked.

- **In-domain upper bound.** val-B comes from the same FoR-2sec package as train and val-A (same
  generators, probably the same speakers; `dataset_audit.md` §8 [doc]). These values describe the
  policy in-domain and are an upper bound, not an estimate of external behaviour; the shortcut
  explanations of [cnn_selection.md](cnn_selection.md) §8 apply unchanged [hypothesis there].
- **CNN vs baseline on val-B is one clip, not a ranking.** CNN val-B EER ≈ 1/669 (above);
  baseline-v1 val-B EER 0.0 ([baseline.md](baseline.md) §4) [doc]. The difference is one spoof
  clip, which val-B cannot separate from chance. val-B set the baseline's threshold and the CNN's
  policy, and neither model was selected on it. The pre-registered comparison is H1 on the FoR test
  split (Stage 11).
- **The error is in FAKE_other.** On val-B the only spoof clip below the bonafide range is
  FAKE_other (§3, item 1). This is a description of one clip in a 100-clip subgroup; it does not
  test anything. H5 (subgroup recall) is tested at Stage 11 only.
- **Where the band falls.** With the threshold set at the EER point and the band set by the 10%
  deferral cap (both pre-declared), the band falls almost entirely on the spoof side of the
  bonafide scores (§3, item 3). This describes what the pre-declared rules did on val-B; nothing in
  the policy follows from it ([risk_run.md](../risk_run.md) Why (e)).

## 5. Environment finding (before the run; no score computed)

Reported by Donia from the Colab session [doc: Donia's account; the notebook output is not
committed], except where marked.

- **CPU runtime.** On a Colab CPU runtime (Intel Xeon @ 2.20 GHz, no AVX-512, Python 3.13.18) the
  val-A log-Mel cache built by notebook 05 had `npz` SHA-256
  `9fbb171196471ca25a23659fcd8d3aafb3e561a50f808cb5a36f5690e98ae19f`, not Stage 8's
  `4502eea2f77cfd83d7e2b1d777e392fa460a13e805a936faa2788df8e7862c81`
  ([stage8_cnn/run_report.json](../audit_evidence/stage8_cnn/run_report.json) [verified]).
  The runner was not started there, so no score was computed.
- **T4 runtime.** The T4 session ran on an Intel Xeon @ 2.00 GHz with AVX-512 [doc: Donia's
  account]. It had Python 3.13.16, torch 2.11.0+cu130 and a T4 GPU, as recorded for Stage 8
  [verified: `stage8_cnn/stage8_export_manifest.json` `python`, `torch`, `gpu_name`]; the Stage 8
  processor was not recorded. There the val-A cache matched `4502eea2…` exactly [verified:
  `run_report.json` `provenance_checks`], so the official run used the T4 runtime. Scoring still
  ran on CPU (`scoring.device` "cpu") [verified].
- **Cause not separable.** The two runtimes differ in processor (instruction set) and in image
  (Python 3.13.18 vs 3.13.16, and possibly other preinstalled packages). The cause may be the
  processor [hypothesis] or the image drift [hypothesis]; this run cannot separate them. scipy,
  soxr, numba and llvmlite are not pinned in `requirements.txt` or `requirements-cnn.txt`
  [verified], and Stage 8 did not record their versions (`stage8_cnn/run_report.json`
  `environment`) [verified].
- **Rule followed.** The switch was made before the run cell, under the rerun rule pre-declared in
  notebook 05 (section 9: only failures before that cell may be rerun from the same commit) [doc].
  Without the switch the runner's `val_a_cache_npz_sha256` check would have stopped the run before
  val-B [derived from [risk_run.md](../risk_run.md) step 3].
- The CPU-runtime `npz` is kept on Drive at `scratch/stage10_diag/` (not committed) [doc].

## 6. Limitations and follow-ups

1. **Validation only.** All values are on val-B, the split the policy was fitted on; they are not
   test results and not external results.
2. **Small-sample descriptive values, no CIs.** One clip is 1/669 ≈ 0.15% of spoof or
   1/660 ≈ 0.15% of bonafide; the D4 confidence intervals are computed on the test split only
   (Protocol §5) [doc].
3. **Stage 11 runtime.** Stage 11 must plan its runtime (the same T4 image family) and how the
   environment is recorded before any FoR test access, so the test cache can be bound to the
   checkpoint's inputs (§5).
4. **Unpinned feature libraries.** Pinning scipy, soxr, numba and llvmlite is a separate, later
   decision; nothing is changed here.
5. **No change follows.** The policy stays as fitted; Stage 12 freezes `risk_policy.json` into F4
   (Protocol §6) [doc].
