# 0012. Model v2 track and one joint external run

- **Status:** Approved (all four; PR #111, 2026-11-10)
- **Date:** 2026-11-09
- **Register:** D48 (amends D36 and D41)
- **Deciders:** Donia Mabrouk (Part C owner, author), Iheb Zemzemi, Malak Ben Salem, Khalil Dridi
- **AI assistance:** Claude (claude.ai, Claude Opus 5.5) drafted this ADR from the repository
  documents and the professor's feedback; Donia reviews it before push.

## Context

- After the midterm review (2026-11-09) the professor asked for a stronger core model ("move to
  LSTM or models specific to voice"), training augmentation for real-world audio, live analysis,
  production-grade infrastructure and a redesigned interface. Model v1 (one CNN, D42) is selected
  and its risk policy is fitted (Stages 8–10 closed); Stage 11 has not run, so **no test result
  exists yet** for any model.
- DEEP-VOICE is touched exactly once (AGENTS.md hard rule 1, ADR 0005). D35 makes the Stage 13
  claim **dataset-wide**, and D36 scores every frozen model in that single pass. If Stage 13 ran now
  with v1 only, D35 would forbid ever scoring a later model on DEEP-VOICE, so v1 and v2 could never
  be compared on the same untouched external data.
- The FoR test split is protected by the run registry (D31, key amended by D41): one `stage11_test`
  claim per model set. A different model set may be registered, but a design chosen after seeing a
  test result would be shaped by that result.

## Decision

Keep v1 on its current path to F4, build a pre-declared model v2 (D49, D50), hold Stage 13 until v2
is frozen, then run DEEP-VOICE **once** on v1, v2 and `baseline-v1` together. The v2 design is
approved **before** the v1 Stage 11 claim, the deployed model is fixed by a rule that uses no test
result, and a cut-off date protects the schedule. Full text: D48 in [`README.md`](README.md).

## Consequences

- **Easier:** a clean, paired external comparison of v1 and v2 (H8) on data no model has seen; the
  real v1 model can replace the `MockAnalyzer` as soon as F4 is verified, so the platform becomes
  testable without waiting for v2.
- **Committed:** the v2 numbers (D49, D50, D52) go into a Protocol addendum (§§4, 6, 7, 9, 12) that is
  merged before the v1 Stage 11 claim; after that claim, any change to them is a post-hoc change and
  needs a logged decision and the label "exploratory" (Protocol §12 change control (3)).
- **Committed:** if F5 is not verified by 2026-12-04 23:59 Africa/Tunis, Stage 13 runs with v1 and
  `baseline-v1` only (D36 as written) and v2 never gets a DEEP-VOICE result. This is a schedule rule,
  fixed before any result.
- **Committed:** v2 is the deployed model once F5 is verified, whatever its Stage 11 or Stage 13
  results. Choosing the deployed model from test results would turn the test sets into selection
  sets.
- **Harder:** the Stage 13 runner (D35, still to be implemented and reviewed) must load three model
  sets and write one stored-score file per model; the Stage 11 runner (PR #68) and the FoR-test
  governance test must accept the v2 model set as a second, separately registered `stage11_test`
  claim; Stage 14 gets a third model set.
- **Risk:** Khalil's Stages 13–16 start later. Mitigation: Stages 14–16 on FoR data can be prepared
  on v1 outputs while v2 trains; the cut-off bounds the delay.

## Options considered

1. **Run Stage 13 now with v1; test v2 later on a second external dataset.** Rejected: no shared
   external set, so no paired v1–v2 comparison; a second external dataset needs its own audit,
   licence check, pinning and governance; the result would mix "better model" with "different data".
2. **Skip v2.** Rejected: it leaves the professor's core feedback unaddressed and keeps a model trained
   only on clean studio audio for a call-centre and messaging use case.
3. **Pause v1 at Stage 10 and wait for v2.** Rejected: the platform would stay on mock data for
   weeks, which is the professor's first criticism.

## Links

- [Decision register](README.md): D31, D35, D36, D41, D46 (PR #67), D48–D52; H8.
- [ADR 0005](0005-deep-voice-single-use.md), [ADR 0007](0007-stage13-reporting-and-access.md),
  [ADR 0009](0009-baseline-in-external-run.md), [ADR 0010](0010-stage14-purpose-key-and-paired-scores.md),
  [ADR 0013](0013-model-v2-candidates-and-augmentation.md).
- [Experimental Protocol](../experimental_protocol.md): §§7, 8, 9, 12.
