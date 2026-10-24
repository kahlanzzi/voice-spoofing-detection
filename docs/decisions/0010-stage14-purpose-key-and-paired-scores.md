# 0010. Stage 14 robustness runs: registry purpose key and paired clean scores

- **Status:** Proposed
- **Date:** 2026-10-24
- **Register:** D41 (amends D31)
- **Deciders:** Donia Mabrouk, Iheb Zemzemi, Malak Ben Salem, Khalil Dridi

## Context

D31 prevents repeated FoR test-split evaluations by keying the append-only registry to a model set.
Stage 14 needs to evaluate pre-declared degradations after Stage 11 has already scored the test
clips. A free-form purpose field would let a caller invent another name and bypass the one-run guard.
Stage 11 also needs to retain clean per-clip scores so Stage 14 can compare paired observations
without scoring clean clips again. D36 identifies the baseline artifact in F4; the ModelBundle
records the CNN bundle seed and all three training seeds.

## Decision

Propose amending D31's key to `(dataset, split, evaluation_purpose, model_set)`, with exactly two
code-defined purposes: `stage11_test` and `stage14_robustness`; refuse every other value, and require
a new decision to add a purpose. Stage 11 stores clean per-clip scores for `baseline-v1` and all three
CNN seeds. Stage 14 makes one run per frozen model set—the `baseline-v1` artifact
`42cc4d53a7cd881f56e4fa54fcf4797de8885b404d0a2b31c293555c868df10a` and the ModelBundle CNN bundle
seed (F4)—over the complete §9 degradation grid, reusing Stage 11 clean scores and pairing by clip
ID. Use 2,000 D14-group bootstrap resamples with seed `2026`; the baseline's 3.4 kHz EER increase is
meaningful only at ≥5 absolute percentage points with the paired 95% CI excluding zero.

## Consequences

- Stage 11 stores stable clip ID, label, D14 group ID and model score outputs; the baseline stores
  raw score and D29 forced-binary prediction, with calibrated probability marked not applicable.
- Stage 14 does not make a second clean pass, fit or tune models, recalibrate, change thresholds, or
  select models. It applies degradations before the unchanged F3 profile.
- The existing H6 CNN endpoint and §9 degradation levels remain unchanged.
- No FoR test access for this comparison until all four approve D41 and the registry and score-storage
  implementation is reviewed. Updating Protocol §§7/9 text is likewise gated on approval.
- Adding another evaluation purpose requires a new decision; unrecognized purpose values fail closed.

## Links

- [Decision register](README.md): D31 and proposed D41.
- [Experimental Protocol](../experimental_protocol.md): §§7, 9 and 14.
- [Evaluation harness](../evaluation_harness.md): run registry and paired bootstrap convention.
