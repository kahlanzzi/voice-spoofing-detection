# 0007. Stage 13 reporting conventions and guarded access

- **Status:** Approved — not permission to run (all four deciders; PR #26 and PR #29, 2026-10-19)
- **Date:** 2026-10-18
- **Register:** D33–D35 in `README.md` (approved)
- **Owner:** Khalil Dridi (D26)
- **Deciders:** Donia Mabrouk, Iheb Zemzemi, Malak Ben Salem, Khalil Dridi; all four accepted
- **AI assistance:** Codex drafted this proposal from repository documents at Khalil's request.

## Context

[doc] Protocol v1.0 §8 fixes 2 s non-overlapping windows, dropping the short remainder;
filtered primary and unfiltered sensitivity views; eight source-speaker rows; and one
guarded external run after the complete ModelBundle freeze. D4 defines metrics, D5 the
secondary file mean, D17 the frozen speech filter, and D19 the secondary speaker average.
Protocol §12 requires decisions before the affected data are seen. Three details remain
open in the project log: undefined speaker averages, file-level decisions and governance access.

[verified from code] `speaker_average` requires an explicit `propagate` or `raise` policy.
It does not enforce the eight-speaker roster. `three_way_decisions` defines the shared
boundary comparisons. The governance test scans repository text for path patterns; it
does not prove that arbitrary code cannot read audio or enforce the single-run rule.

## Proposed decisions

### D33 — Undefined equal-speaker averages

For each view separately, report the unweighted arithmetic mean of each rate across the
eight expected source speakers only when all eight rates are defined. Otherwise report
the mean as undefined (`NaN` internally, `NA` in tables, `null` in strict JSON), with
`n_speakers=8`, `n_defined`, affected speaker IDs and the reason each value is undefined.
Use the existing `propagate` option; keep `raise` as a development validation option.
Never omit a speaker, substitute zero, or average only the defined subset.

Build the expected roster from the frozen metadata manifest. Validate that each retained
row has a known source speaker and one consistent label. Unexpected/missing score records,
duplicate windows and inconsistent labels are integrity errors, not undefined metrics.
A speaker with no retained windows due to the frozen filter still gets a table row:
retained count zero; EER/AUC and decision rates undefined; coverage undefined because its
retained-window denominator is zero. Keep its drop counts and filter-retention rate in a
separate report. Filter rejection is not model deferral. Single-class EER/AUC and zero-
denominator rates remain undefined as in the harness. An entirely empty view is explicitly
reported as unevaluable; do not fabricate rows for the current helper or silently skip it.

The reporting layer must implement the zero-window case and roster checks before use;
the current helper rejects empty input. Counts and confusion matrices stay as counts,
not rate averages. Primary pooled metrics remain separate. Use the existing `bootstrap_ci`
convention ([evaluation harness, Step 4](../evaluation_harness.md#step-4-bootstrap-confidence-intervals),
project-log Entry 31): resamples whose statistic raises `ValueError` or returns NaN are
excluded from the percentile calculation and counted in `n_invalid`. Report `n_invalid`
with every interval, including zero. D11/Protocol §5 still govern clustering, seed and
resample count. An undefined full-data estimate or no valid resamples makes the interval
unavailable; report that fact and reason, not a fabricated interval. This does not silently
omit speakers from the full-data mean under D33.

**Reason / alternatives:** Propagation preserves the intended eight-speaker denominator
and makes missing information visible. Omitting NaNs changes the population; zero filling
invents a result. Aborting all reporting with `raise` hides other metrics that remain valid.

### D34 — Secondary file-level decisions

For the frozen CNN, compute each file's arithmetic mean of calibrated window probabilities
`p` from stored scores. Average probabilities after the frozen calibration, not raw scores
followed by calibration. Compute a filtered file view from retained windows and an
unfiltered sensitivity view from all complete windows of that file; never mix views.
Use the same frozen threshold `t` and half-width `h` as the window-level policy:

- Forced binary: spoof when mean `p >= t`, otherwise bonafide.
- Three-way: synthetic when mean `p >= t+h`; authentic when mean `p < t-h`;
  uncertain otherwise. Reuse the shared decision function, including exact boundaries.

Do not fit file thresholds, recalibrate file means, optimize the band, average hard labels,
or use majority voting. File means are secondary scores, not a claim that calibration
remains valid after aggregation. Do not copy the baseline's raw-score threshold onto CNN
probabilities or invent calibrated baseline outputs (D29).

Each evaluable file contributes once to file-level metrics, using its manifest label.
Report expected/evaluable/no-window file counts per class and source speaker. A file with
zero windows for a view has an undefined mean and no model prediction; list it explicitly,
exclude it from score/decision metric denominators, and report that exclusion. Never label
it authentic or uncertain merely because data are absent. If a view has no evaluable files,
its metrics are undefined. Per-speaker summaries still retain all eight expected speakers
under D33. File-level CIs retain source-speaker clustering, not independent file resampling.

Derive both views from the one stored-score output; later analysis must not rerun the model
or reopen audio. Validate bundle/run identity, complete window indices, finite probabilities
in [0,1], labels and manifest membership before aggregation. No new score columns or
changes to the frozen C→D interface are approved by this proposal.

**Reason / alternatives:** This fills D5's decision-level gap using the existing policy.
Majority vote differs from a mean score; tuning a file threshold on external labels would
violate the external-test rule.

### D35 — Exact guarded-run allowlist additions

Propose only these two new exact entries in the governance test's `ALLOWED` tuple:

| Proposed file (not implemented by this PR) | Purpose |
|---|---|
| `src/evaluation/external_run.py` | Sole Stage 13 orchestration entry point for manifest-driven, guarded scoring and saved output |
| `configs/external_run.yaml` | Configuration consumed by that entry point; paths supplied through config/CLI, no secrets |

Keep existing allowed entries. Do not add directory prefixes such as `src/evaluation/`,
`configs/` or `notebooks/`, and do not weaken path-pattern scanning. No Stage 13 notebook
exception is proposed. If implementation needs a different or additional path, return for
an explicit decision amendment before changing the allowlist.

Approval of names is not authorization to execute. A later reviewed implementation PR must
add the exact entries and synthetic tests proving lookalike paths and unrelated loaders
are rejected. It must also demonstrate, using dummy files and a fake scorer:

1. Verify the approved frozen bundle, profile/policy versions and pinned manifest/hash ledger
   before any audio decoder or model call. Refuse mismatches and incomplete prerequisites.
2. Key the external single-run claim by the dataset: one external run ever, for any model
   set. Do not use the Stage 11 model-set key. A new run ID, output folder, model hash or
   model set must not create a second opportunity. Make the claim atomic and durable beyond
   the Colab session: commit it to the repository, or write it to Drive and verify it,
   before the first audio read or model call. A session-local claim alone is insufficient;
   abort before scoring if durable persistence cannot be confirmed. Test simultaneous starts,
   second-start refusal across model sets and loss of the local session.
3. Persist failed/incomplete claims as well as successful ones. No automatic retry or resume
   of a partially executed real run; retain evidence and stop for team review. This does
   not authorize a second run.
4. Immediately after the run, record the registry SHA-256 and line count in the project
   log, preserving the durable registry evidence (D31 truncation lesson).
5. Produce both views in one scoring pass with drop counts and the existing stored-score
   interface; subsequent reporting consumes scores only. Keep D27's no-MCP Stage 13 rule.
6. Add a runner integration test proving that every complete 2 s window calls the same
   frozen Stage 6 preprocessing function with the frozen profile and parameters. Trimming
   and centred padding must happen separately per window, not on the full recording before
   windowing. Use synthetic multi-window audio with distinct silence patterns and a fake
   scorer; wrap the real preprocessing function to verify call inputs/counts and compare
   each scorer input with a direct call on that individual window. Choose fixtures that
   fail if trimming/padding is done before splitting, and check the frozen padding convention
   and protocol's short-remainder rule. Use the final F3 API; do not duplicate preprocessing
   logic or access external audio. This test is required implementation work, not yet built.

Reusing the Stage 11 registry requires inspection and tests: its model-set key must not
be used for the external guard, whose uniqueness scope is the dataset across all model sets. The current governance
scanner is a text guard, not a security sandbox; review actual I/O and runtime guards too.
Existing D20 metadata-only permissions remain the only pre-run dataset access permissions.

## Separate follow-up: external model comparison

Donia identified a separate question: Protocol §1 RQ2 compares CNN and baseline externally,
while the currently described external runner scores the CNN bundle. She plans a separate
decision proposing both frozen models on the same windows under one claim. This ADR does
not approve that scope change; resolve it before implementing or executing the external run.
D34 remains unchanged and describes CNN file-level decisions only.

## Consequences and review gate

These proposals make undefined outputs and denominators explicit, reuse the frozen policy,
and bound the planned access exception. Costs: additional completeness/reporting tests and
a guarded orchestration implementation remain necessary. No external data were read and no
model was run in preparing this document.

All four reviewers should explicitly accept or revise D33, D34 and D35 separately. Record
the approval date and PR for each register row; do not infer agreement from silence or CI.
Only after acceptance, update the protocol change log and affected documentation through
the agreed change-control process. Allowlist edits and runtime implementation require their
own reviewed code/test changes. F3, real Part C work, F4 and the Stage 13 review gate still apply.

| Reviewer | D33 | D34 | D35 | Date / evidence |
|---|---|---|---|---|
| Donia Mabrouk | ✅ | ✅ | ✅ | 2026-10-18, PR #26, after her requested changes were incorporated |
| Iheb Zemzemi | ✅ | ✅ | ✅ | 2026-10-19, approved PR #29 (GitHub review), which asked him to accept or revise D33–D36 |
| Malak Ben Salem | ✅ | ✅ | ✅ | 2026-10-18, PR #26 |
| Khalil Dridi | ✅ | ✅ | ✅ | Author, PR #26 |

## Links

- [Decision register](README.md): D2, D4–D5, D11, D17, D19–D21, D26, D29, D31.
- [Protocol](../experimental_protocol.md): §§5–8, 11–12.
- [Speaker average](../speaker_average.md) and [per-speaker tables](../speaker_analysis.md).
- [Governance test](../../tests/governance/test_deepvoice_access.py).
- [Shared decisions](../../src/risk/decisions.py).
