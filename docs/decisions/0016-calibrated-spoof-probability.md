# 0016. Expose calibrated spoof probability and policy context

- **Status:** Proposed — awaiting all-four team approval
- **Date:** 2026-11-15
- **Register:** D63
- **Author:** Khalil Dridi
- **Deciders:** Pending; no implementation approval is recorded yet.
- **API-owner confirmed; all-four approval pending:** zero kept windows return `422 no_speech`
  with no stored analysis; windows use a versioned JSON array (JSONB on PostgreSQL).
- **Pending team decision:** how uploads with only one or two kept windows are handled. This draft
  recommends applying D34's mean rule whenever at least one window is kept; D52's three-window
  warm-up remains live-only.

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

Amend D26's C-to-E response and D55's unchanged-API clause as one coordinated contract update. This
section is a proposal only until all four members approve D63 and the approval/status commit is merged.

- **Probability and policy identity:** Each scored window's `p` is the unrounded calibrated probability
  that the window is synthetic, from the calibration component of the immutable policy identified by
  the same response's `policy_version`. Upload `spoof_probability` is D34's mean of kept-window `p`
  values. Apply the frozen policy to that exact unrounded mean. For live analysis, each update carries
  the window `p`; the running verdict applies the same rule to the D52 mean of kept windows. Do not
  round before aggregation or policy evaluation. This also applies to F5 and later policy versions.
- **Verdict and confidence:** Use `three_way_decisions` directly on a window `p` or the file's unrounded
  mean `p`. Do not call `RiskPolicy.decide()` on already calibrated `p`, because it calibrates again.
  `confidence = max(p, 1 - p)` rounded to two decimals, computed from the same unrounded file mean.
  Keep confidence separate; never reconstruct `p` from it.
- **Policy values and provenance:** Return the verified policy's `threshold`, server-computed
  `uncertain_band`, and provenance from the same `policy_version`. The band is
  `[threshold - half_width, threshold + half_width]`; the server returns both exact bounds so the
  client does not derive them. The server applies `three_way_decisions`: authentic when
  `p < lower`, uncertain when `lower <= p < upper`, and synthetic when `p >= upper`.
  `policy_provenance` has typed fields: `threshold: number`, `half_width: number`,
  `target_deferral: number`, `deferral_on_fit: boolean`, `n_spoof_fit: integer`,
  `n_bonafide_fit: integer`, `model_version: string`, and
  `bundle_manifest_sha256: 64-character lowercase hex string`. Copy them from the verified
  `RiskPolicy` and bundle manifest; never hand-type or infer them. `RiskPolicy` has no EER-point
  field; do not claim one in provenance.
- **Upload view and window shape:** Use D17's filtered view for scoring and the file-level mean; the
  unfiltered sensitivity view remains research-only. Include every complete non-overlapping two-second
  window in chronological order, including windows dropped by D17. `index` is zero-based;
  `start_s` is inclusive and `end_s` exclusive, both seconds from the start of the uploaded audio.
  Each entry is `{index: integer, start_s: number, end_s: number, p: number | null,
  label: synthetic | uncertain | authentic | no_speech}`; `p` is required and never omitted.
  A kept window has unrounded binary64
  calibrated `p` and a server-computed policy label. A D17-dropped window has `p: null` and
  `label: no_speech`; only kept windows contribute to file probability, verdict and confidence. A
  trailing fragment shorter than two seconds is omitted under D5.
- **Zero kept windows:** Return a structured `422 no_speech` error and create no analysis record.
  This is an application-upload rule; it does not change D34's Stage 13 reporting of zero-window
  files. For each such rejection, emit exactly one structured `service_log` event containing
  `request_id` and `error: no_speech`; never include audio bytes, request body or original file name.
  The UI maps this to a new S5 `no_speech` error kind (the 11th S5 kind after D63 approval).
- **One or two kept upload windows (pending all-four approval):** Recommended rule is to return a
  normal upload result using D34's mean over the D17-kept windows, even when that mean has one or two
  terms. D34 defines the upload/file rule and has no three-window minimum; D52's three-kept-window
  warm-up applies only to live updates. This explicit minimum-window rule needs team approval.
- **Live update envelope:** The D52 WebSocket sends one JSON object for each complete two-second
  window. Its exact proposed fields are `type: "window_update"`, `model_version: string`,
  `profile_version: string`, `policy_version: string`, `bundle_manifest_sha256: string`,
  `threshold: number`, `uncertain_band: [number, number]`, `policy_provenance: PolicyProvenance`,
  `window: AnalysisWindow`, and `running_result: RunningResult | null`. The window uses the upload
  shape, including `p` and label even before warm-up; a D17-dropped window has `p: null` and
  `label: no_speech`; `p` is null if and only if the label is `no_speech`. No-speech windows remain
  in the timeline but do not contribute to D52's running mean. `running_result` is null until D52's
  three-kept-window warm-up is met;
  afterward it contains `spoof_probability` (the unrounded running mean), server-computed
  `prediction`, `confidence` (rounded to two decimals from that same mean), and `risk_level`.
  Errors contain no probability, verdict or window. The browser never computes labels or verdicts
  from `p`. The capture lifecycle and audio-retention rule in ADR 0014 are unchanged; live audio
  and per-window values remain in memory, not stored.
- **Persistence (API owner confirms JSON; all-four approval pending):** Store the ordered window array
  with its analysis: JSON text in SQLite and JSONB in PostgreSQL, never as a separate queryable
  child table. Validate on write against schema version 1, with at most 30 entries, the four-label
  enum, and `p` either null or a number in `[0, 1]`; `p` is null if and only if `label` is
  `no_speech`. Store `windows_schema_version` beside the JSON. Serialize and parse numeric values
  with a binary64 round-trip; do not quantize or round them. Add nullable policy columns to the parent:
  `spoof_probability`, `threshold`, `half_width`, `target_deferral`, `deferral_on_fit`,
  `n_spoof_fit`, `n_bonafide_fit`, and `bundle_manifest_sha256`; `uncertain_band` is derived by
  the server from threshold and half-width. Do not store audio or the original file name. A child
  table is only needed if querying windows across analyses becomes a requirement.
- **History/detail:** `POST /analyses` and `GET /analyses/{request_id}` return the detail shape with
  `windows`. `GET /analyses` list items return the same scalar analysis and provenance fields but
  omit `windows`; the History table stays compact. Existing mock results and historical rows whose
  values cannot be recovered return `null` for `spoof_probability`, `threshold`,
  `uncertain_band`, `policy_provenance`, and `windows`. Do not invent values from confidence.
- **Errors and migration:** An upload with zero kept windows returns the structured section 6
  `422 no_speech` error and creates no analysis row (confirmed by Iheb; pending all-four approval).
  Error responses
  carry no verdict, probability, policy values or window result. A live update before warm-up may
  include its individual window `p` and label, but has `running_result: null`. Add one coordinated
  additive migration for the response, stored policy values, JSON window column and its schema
  version; legacy rows remain
  readable and unavailable values stay `null`. Coordinate the base with Iheb before implementation.
- **Scope and versioning:** Do not change calibration, thresholds, risk policy, labels, metrics, or
  Stage 13 gates. Keep D63 as the decision number; D62 is unused because V2b-4 found no all-zero
  clips and will log that result. Keep SRS v4.0 as current. After D63 approval, update SRS v4.1 as a
  synchronized Markdown/PDF pair and update its index/AGENTS pointer only when v4.1 becomes current.

## Consequences

- The API design, Protocol §11, stored result, migration, TypeScript types, upload/live envelopes, and
  result/detail UI must be updated together from one approved base.
- The live and upload displays use the same policy version and boundary rule; the browser does not
  hard-code thresholds or recompute labels.
- Mock and legacy values remain unavailable. No value is reconstructed from rounded confidence.
- This work is separate from V1c and the Stage 13 external run. Stage 13 access and scoring gates
  remain unchanged.
- D63 is **not approved** by this draft. Implementation must wait for all-four approval and the
  merged approval/status commit. SRS v4.0 remains current until v4.1 is reviewed and adopted.

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
