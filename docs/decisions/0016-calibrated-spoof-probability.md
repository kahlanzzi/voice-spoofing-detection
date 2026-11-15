# 0016. Expose calibrated spoof probability in analysis results

- **Status:** Proposed
- **Date:** 2026-11-15
- **Register:** D63 (provisional; D62 remains reserved; renumber if needed before merge)
- **Deciders:** Pending team review

## Context

SRS v4.0 FR-06 requires calibrated probability `p` where applicable, and D55's decision board calls for the calibrated `p`, the frozen threshold and uncertain band. D55 also says the API contract remains unchanged. The current API contract defines an exact eight-field `AnalysisResponse` with `confidence`, but no probability field; `analysis_record` likewise stores no `p`.

The risk-policy documentation defines confidence as `max(p, 1 - p)`, rounded to two decimals. Confidence is a certainty measure, not a full-precision directional probability. Although the current policy can constrain the possible range for some verdicts, the API does not promise that clients can reconstruct the original calibrated `p` from rounded confidence across all results. Clients should not reverse-engineer policy internals to do so.

This creates a conflict among SRS v4.0 FR-06, D55's decision-board requirement and D55's unchanged-API clause, as well as the Stage 3 C-to-E interface recorded in D26. Under `AGENTS.md`, that conflict must be resolved by an explicit team decision rather than silently changing the contract.

## Proposed decision

Amend D26's C-to-E response and D55's unchanged-API clause to add `spoof_probability` to successful analysis results and stored analysis responses:

- A numeric value in `[0, 1]` is the analyzer's calibrated probability that the recording is synthetic. The server continues to apply the frozen, versioned policy to the full-precision value.
- Keep `confidence = max(p, 1 - p)` unchanged. Return both values; do not derive one from the other in the client.
- Return `null` when no calibrated probability is available, including mock results and historical records without an original probability. Do not invent or backfill a value from rounded confidence.
- Persist the probability for new real-model results. Existing records remain readable with `null`; errors and incomplete live updates still carry no result probability.
- The UI identifies mock and unavailable values explicitly. A displayed mock value must never be described as calibrated model evidence.

This is a proposal only. No API, SRS, database or UI contract is changed by this ADR until the team approves it.

## Consequences if approved

- The API, stored-result shape, TypeScript types and any completed live-result envelope need a coordinated update.
- The database change needs a migration that preserves old records and leaves unrecoverable probabilities unknown.
- SRS v4.0 Markdown and its PDF must be updated and regenerated together after approval. The API design and decision board documentation must be aligned in the same implementation/revision plan.
- The model, calibration fit, risk thresholds, uncertain band, metrics, labels and Stage 13 gates remain unchanged.

## Alternatives considered

- **Keep the API unchanged and use `confidence` as `p`:** rejected because confidence has different semantics and is rounded.
- **Redefine confidence to mean spoof probability:** rejected because it silently changes the frozen response contract and existing interpretation.
- **Expose only the raw model score:** rejected because D55 and SRS v4.0 call for calibrated probability.

## Links

- [Decision register](README.md) (proposed D63)
- [SRS v4.0](../reference/SRS_v4.0.md), FR-06 and FR-08
- [API design](../api_design.md), `AnalysisResponse` and `analysis_record`
- [Risk-policy fit](../results/risk_policy_fit.md), confidence interpretation
- D26 (C-to-E interface) and D55 (analyst interface)
