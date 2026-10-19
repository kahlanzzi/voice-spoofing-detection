# 0008. Baseline in the single external run

- **Status:** Approved — not permission to run (all four deciders; PR #29, 2026-10-19)
- **Date:** 2026-10-18
- **Register:** D36 in `README.md` (approved)
- **Owner:** Donia Mabrouk (Part C, D26)
- **Deciders:** Donia Mabrouk, Iheb Zemzemi, Malak Ben Salem, Khalil Dridi; all four accepted
- **AI assistance:** Claude Code (Claude Opus 5.5) drafted this proposal from repository documents
  at Donia's request; Donia reviews before push.

## Context

[doc] Protocol v1.0 §1 defines RQ2 as "CNN vs baseline, in-domain and externally". The in-domain
half is tested in Stage 11 (H1, FoR test). The external half has no run: §8 scores only the frozen
CNN ModelBundle ("exactly one run of the frozen bundle"), and §11 defines the C→D stored scores as
one per-window CSV for that bundle. As written, the external half of RQ2 cannot be answered.

[doc] Constraints on any fix:
- Hard rule 1, D20 and D35: the external set is touched exactly once; D35 keys the single-run
  claim by the dataset, for any model set, so a later baseline run is impossible by design.
- D2 and hard rule 2: everything scored externally is fitted on FoR only and frozen before the run.
- D7 and D29: the baseline (MFCC mean + std, RBF-SVM) is evaluation-only, with its own val-B EER
  threshold, forced binary only; no calibration, no uncertain band.
- D11 and Protocol §5: bootstrap 95% CIs clustered by source speaker for the external set.
  [verified from code] `bootstrap_ci(statistic, n, clusters=...)` takes a statistic of indices, so
  one index set per resample feeds both systems of a paired difference
  (`docs/evaluation_harness.md`, Step 4, reason (a)).
- D26: changing the C→D interface after Stage 3 needs a decision.

The gap was found during the review of the Stage 3 brief (project log, Entry 47) and noted as a
separate follow-up in ADR 0007. No external data and no model output have been seen.

## Decision

### D36 — Both frozen models are scored in the one external run

1. **One run, both models.** The single Stage 13 run scores both frozen models, the CNN
   ModelBundle and the MFCC/SVM baseline, on the same external windows, in the same pass, under
   one claim (the D35 dataset-wide key). There is no second run for either model.
2. **Baseline reporting.** The baseline uses its own frozen D29 threshold, forced binary only,
   plus EER and ROC-AUC on its continuous score. No calibration, no uncertain band, no risk level
   (D7, D29). The same filtered and unfiltered views (D17), the per-speaker table (D19) and the D33
   conventions apply. D34 (file-level decisions) stays CNN-only, as ADR 0007 states.
3. **Paired comparison.** Report the CNN − baseline EER difference on the same windows, with a
   speaker-clustered paired bootstrap 95% CI: `bootstrap_ci` with `clusters` = source speaker, one
   index set per resample used for both models (D11, Protocol §5). The 8-cluster limitation and
   `n_invalid` are reported next to the interval (evaluation harness, Step 4; D33).
4. **Freeze.** The baseline artifact (SVM model, MFCC settings from the F3 profile, D29 threshold)
   is hashed, and the hash is recorded in the F4 freeze decision together with the bundle's
   `manifest_sha256`. The baseline is **not** added to the deployment ModelBundle (evaluation only,
   D7); the API never loads it.
5. **C→D interface (D26), amended.** One stored-score file per model, with identical columns
   (Protocol §8). In the baseline file: raw score = the SVM's continuous score; calibrated p = NA
   (empty cell); prediction = the forced-binary label at its D29 threshold, only "synthetic" or
   "authentic", never "uncertain". All other columns are filled as in the CNN file. Nothing else
   in the interface changes.

## Alternatives considered

- **Drop the external half of RQ2.** Allowed by Protocol §12 change control (no data seen), but
  rejected: it removes the external reference point that makes the CNN's generalization result
  interpretable.
- **A separate baseline run.** A second touch of the external set, forbidden by hard rule 1 and
  by the D35 dataset-wide claim; rejected.

## Consequences

- The Stage 13 runner (D35) must load and verify **both** artifacts (bundle manifest and baseline
  hash) before any audio read or model call, and refuse to start if either is missing or mismatched.
- F4 records two artifacts: the ModelBundle `manifest_sha256` and the baseline artifact hash.
- Stage 13 produces two stored-score files; all later reporting reads scores only, never audio.
- No change to the deployed system, the risk policy, D29, D33–D35 or the Stage 11 rules.
- To verify later: the runner tests (synthetic data, fake scorers) show that both artifacts are
  checked before any audio read and that both files come from one claim.

## Review

All four deciders accept or revise D36 explicitly before F4. Record the date and PR in the
register; do not infer agreement from silence or CI.

| Reviewer | D36 | Date / evidence |
|---|---|---|
| Donia Mabrouk | ✅ | Author, 2026-10-18 |
| Iheb Zemzemi | ✅ | 2026-10-19, approved PR #29 (GitHub review) |
| Malak Ben Salem | ✅ | 2026-10-19, approved PR #29 (GitHub review) |
| Khalil Dridi | ✅ | 2026-10-19, approved PR #29 (GitHub review) |

## Links

- [Decision register](README.md): D2, D4, D7, D11, D17, D19, D20, D26, D29, D33–D35.
- [ADR 0007](0007-stage13-reporting-and-access.md): Stage 13 reporting and guarded access.
- [Protocol](../experimental_protocol.md): §1 (RQ2), §5, §8, §11, §12.
- [Evaluation harness](../evaluation_harness.md), Step 4 (bootstrap, paired differences).
- [ModelBundle](../model_bundle.md) (`manifest_sha256`, F4).
