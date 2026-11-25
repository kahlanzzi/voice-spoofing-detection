# 0018. Analyst interface: VoiceCheck Enterprise as the D55 design

- **Status:** Approved 2026-11-25 (Iheb, Malak and Khalil in person, user-reported; reviews on PR #167 by Iheb (`kahlanzzi`) and Khalil (`dridi-khalil`); recorded in Entry 220)
- **Date:** 2026-11-25
- **Register:** D67 (amends D65)
- **Author:** Donia Mabrouk
- **Deciders:** Iheb Zemzemi, Malak Ben Salem, Khalil Dridi (approvers); Donia Mabrouk (author)

## Context

- D55 (1) has the team choose one design direction. D67 refines D55 (1) again; D55 (2)–(4) unchanged.
- D65 ([ADR 0017](0017-ui-direction-d.md), approved 2026-11-17) chose Direction D, specified in HANDOFF-v2.
- PR #166 added Malak's VoiceCheck Enterprise prototype as a candidate (project log Entry 218).
- Entry 218's review issues: the candidate uses a median for live scores and excludes three warm-up windows
  from scoring, while D52 requires the mean of all kept windows; its score scale omits the prominent 0.633
  cut, and its Inter typography differs from D65. Its new Overview, review and admin screens need separately
  defined backend and privacy behavior.

## Decision

The team adopts Malak's prototype `docs/design/voicecheck-enterprise-proposal.html` (SHA-256
`390f35c885cf6d84d80fc663d7fb8dfe8a95c689760e9b09a28a6fd8fd5dacd6`, PR #166) as the D55 build target,
replacing Direction D (its prototype and HANDOFF-v2 stay as history). This decision amends D65 and supersedes
[ADR 0017](0017-ui-direction-d.md).

Screens: Overview, Analyse, Result, Live, History, Flagged reviews, Administration (Members).

### Corrections required before build

The prototype is not followed where it differs:

1. live aggregate = **mean of every kept window** (D52); the first three kept windows count in the mean, only
   the verdict is withheld during warm-up (no median, no "unscored" windows).
2. the decision board keeps D55 (2) and D65 (1): calibrated `p` with the frozen threshold and uncertain band
   drawn, and **0.270 / 0.633 / 0.997 as major labels at every width**; the prototype's nonlinear scale is
   kept and is **labelled as nonlinear on screen**.
3. policy numbers, versions and windows come from the API (D63), never from constants in the web app, with one
   exception until D63 ships: results use the "Score not available" state, whose zone edges come from the
   HANDOFF-v2 §5.9 display table keyed by `policy_version` (`risk-v1` → threshold 0.633, band 0.270 to 0.997);
   the table is removed when D63 ships.
4. the original audio filename is never stored (`docs/api_design.md`: "No audio, original file name or IP
   address is ever stored (FR-09, NFR-04)") and is no longer shown either (a change from HANDOFF-v2 §5.13 and
   the shipped app): the request ID identifies a result.
5. fonts **Public Sans + IBM Plex Mono** (as D65), **self-hosted** (the CSP `default-src 'self'` forbids
   external font URLs); IBM Plex Mono for numbers and IDs only.
6. the "Explore demo" sign-in button, invented data and the DEMO badge do not exist in the built app.
7. the D65 colour rule stays: a verdict colour is never the only signal.

### Build and test rules

1. **Read-side API:** Overview uses the existing `GET /api/stats` (§4.7, no change); Flagged reviews =
   synthetic or uncertain results, Pending (no note) and Reviewed (with note), served by a new History filter
   `note=missing|present` on `GET /api/analyses` (§4.4; additive, no migration); notes stay §4.6.
2. **Kept from D55/D65:** §11 fields, NFR-05 decision-support wording, axe e2e (WCAG 2.0–2.2 A/AA, light and
   dark), 390 px checks, colours only in `tokens.css`, the D55 (4) usability test and SUS ≥ 68 with its task
   script unchanged (no flagged-review task).
3. **Handoff:** A new `docs/design/HANDOFF-v3.md` specifies the screens against the prototype with these
   corrections.
4. **Live check screen:** built against the `/api/live` contract in SDD v2.0's streaming section (messages,
   window timing, D52 mean of all kept windows, verdict withheld during warm-up, error cases; due Sun 29 Nov)
   and a mock, from Mon 30 Nov.
5. **Sign-in for the usability test:** the current D47 username sign-in; it switches to email only when D68 is
   merged.
6. **Usability test date:** W5 window Fri 11 – Sun 13 Dec if L1 is on staging by Thu 10 Dec; otherwise Mon 14
   – Tue 15 Dec.

### Reason and options rejected

- **Reason:** the team judged the Enterprise direction a better fit for the analyst's decision workflow and
  accepts the extra build, shared across the team.
- **Option rejected:** keeping Direction D (decided by the team, 2026-11-25); a linear 0–1 board scale (team
  answer 25 Nov: nonlinear, labelled).

## Consequences

**Easier:**

- The build is shared across the team.
- The read side reuses `GET /api/stats` (no change) and notes; the new `note=missing|present` filter is
  additive, with no migration.

**Harder or committed:**

- The team accepts the extra build.
- A new `docs/design/HANDOFF-v3.md` is needed to specify the screens against the prototype with the
  corrections.
- Direction D's prototype and HANDOFF-v2 stay as history; the HANDOFF-v2 §5.9 display table gives zone edges
  until D63 ships, then it is removed.
- The original audio filename is no longer shown, a change from HANDOFF-v2 §5.13 and the shipped app.

## Links

D55, D65, D52 and D63 in [README.md](README.md) · [ADR 0017](0017-ui-direction-d.md) · PR #166 ·
[prototype](../design/voicecheck-enterprise-proposal.html) · project log Entries 218 and 219
