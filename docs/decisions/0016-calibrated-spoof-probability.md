# 0016. Expose calibrated spoof probability and policy context

- **Status:** Proposed — awaiting all-four team approval
- **Date:** 2026-11-15
- **Register:** D63
- **Author:** Khalil Dridi
- **Deciders:** Pending; no implementation approval is recorded yet.

## Context

SRS v4.0 FR-06 requires calibrated probability `p` where applicable. D55 (2) requires the decision
board to show calibrated `p`, the frozen threshold and uncertain band, and explain their derivation.
D55 also requires a per-window timeline for uploads and live sessions. D52 (4) requires the live
timeline to show each kept window's calibrated `p` and label. D26 and D55 currently preserve an API
contract that does not expose these values.

The current successful `AnalysisResponse` includes `confidence`, but not calibrated `p`, the policy
threshold, the uncertain band, or per-window values. The database and live event contract have the
same gap. Confidence is `max(p, 1 - p)` rounded to two decimals; it is a different quantity and
cannot be used to recover the directional, full-precision `p`.

Changing only `spoof_probability` would leave D55's decision board and D52's timelines without the
values they require, and could force a second API/database migration during Iheb's integration.
This proposal therefore specifies one coordinated contract change. It does not take effect until
all four team members approve D63 and the approval/status commit is merged.

## Proposed decision

Amend D26's C-to-E response and D55's unchanged-API clause as one coordinated contract update:

- **Probability and policy identity:** Each window's `p` is the unrounded calibrated probability
  that the window is synthetic, from the calibration component of the immutable policy identified
  by `policy_version` in the same response/update. For an upload, response-level `spoof_probability`
  is D34's mean calibrated `p` over kept windows; that full-precision mean is the exact value to
  which the frozen risk policy is applied. For live analysis, each update carries the window `p`,
  and the running verdict applies the policy to the D52 mean over kept windows. Do not round `p`
  before aggregation or policy evaluation. This remains true for F5 and later model/policy versions.
- **Confidence:** Preserve `confidence = max(p, 1 - p)`, rounded to two decimals, with its existing
  meaning. Return it separately; never derive `spoof_probability` from confidence.
- **Decision-board policy values:** Include the frozen `threshold`, `uncertain_band: [lo, hi]`, and
  `policy_provenance` (val-B EER point, 10% deferral cap, and frozen bundle/model identity) belonging
  to the same `policy_version` in successful analysis responses and stored upload results. The UI
  must not hard-code values or infer them from the prediction.
- **Upload timeline:** Include each evaluated window's calibrated `p` and policy-derived label in
  the successful upload result, and persist those values with the stored analysis so the upload
  timeline can be shown consistently from the result and history/detail views. Keep window order and
  the D34 file-level decision separate from each window's label.
- **Live timeline:** Each D52 live update includes the kept window's calibrated `p` and derived
  label, together with enough policy identity/context to render the same timeline. Non-speech
  windows remain represented as “no speech,” not assigned a fabricated probability. Live window
  values remain in memory and are not stored, consistent with D51/D52.
- **Unavailable values:** Return `null` for mock results and historical records whose original
  calibrated/window values were not stored. Do not invent values from rounded confidence. A mock
  result must never be described as calibrated model evidence.
- **Errors:** Error responses and incomplete live updates contain no verdict, probability, policy
  result, or partial window result.
- **Migration:** Make one additive migration for the new response and stored-upload fields. Legacy
  rows remain readable; values that cannot be recovered exactly remain unavailable. Coordinate the
  API and migration base with Iheb before implementation.
- **Policy and research behavior:** Do not refit calibration, move the threshold or band, change
  labels, metrics, or the Stage 13 run/gates. This decision exposes the values from the named frozen
  policy; it does not change that policy or authorize external-data access.
- **Versioning and approval:** Keep D63 as the decision number. D62 is unused because V2b-4 found no
  all-zero clips; V2b-4 will record that result. After D63 is approved, update SRS v4.1 as a
  synchronized Markdown/PDF pair while retaining v4.0 as history. Update the reference index and
  AGENTS.md only when v4.1 becomes the current approved SRS.

## Consequences

- The API, stored-upload shape, database migration, TypeScript types, upload timeline, live event
  envelope and UI must be updated together from one approved base.
- The same immutable policy version governs calibration, threshold and uncertain band; displayed
  values must match the policy used for that analysis, including for F5/v2.
- Mock and legacy values remain explicitly unavailable. No value is reconstructed from confidence.
- This work is separate from V1c and the Stage 13 external run. Stage 13's access and scoring gates
  remain unchanged.
- D63 is **not approved** by this draft. Implementation must wait for all-four approval and the
  merged approval/status commit. SRS v4.0 remains the current approved requirements until v4.1 is
  reviewed and adopted.

## Alternatives considered

- **Add only `spoof_probability` now and add policy/window fields later:** rejected because it would
  leave D55/D52 incomplete and risk a second API/storage migration during integration.
- **Use `confidence` as `p` or reconstruct `p` from confidence:** rejected because confidence is
  rounded and has different semantics.
- **Fetch threshold/band from a separate endpoint:** deferred; D63 proposes a self-contained,
  version-matched response so a result cannot be shown against a different active policy.
- **Expose only the raw model score:** rejected because D55 and SRS FR-06 require calibrated
  probability.

## Approval sequence

1. Revise and review this ADR in the D63 decision PR.
2. Obtain explicit approval from all four team members and merge the approval/status commit.
3. Coordinate the implementation base and API migration with Iheb; implement the full approved
   contract in a separate PR.
4. Update and adopt the synchronized SRS v4.1 pair and make it current only after approval.

## Links

- [Decision register](README.md) (D63; proposed)
- [SRS v4.0](../reference/SRS_v4.0.md) (current pending D63 and v4.1 approval)
- [API design](../api_design.md), `AnalysisResponse` and `analysis_record`
- [Risk-policy fit](../results/risk_policy_fit.md)
- [ADR 0014](0014-live-analysis.md) (D52)
- D26 (C-to-E interface) and D55 (analyst interface)
