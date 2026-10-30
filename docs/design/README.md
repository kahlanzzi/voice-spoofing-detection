# Design: analyst console prototype

`analyst-console-prototype.html` is the clickable prototype of the analyst interface approved in
D38 / [ADR 0008](../decisions/0008-web-frontend-react.md). It is a **design reference for Stage 18,
not production code**: one self-contained HTML file with mock data, no backend and no model.
[`HANDOFF.md`](HANDOFF.md) is the developer handoff: screens, flows, tokens, components,
accessibility notes, open decisions and the verification report.

Version 4 (2026-10-29) is the proposed final design reference. It builds on v3 (second review
round) and follows the fitted `risk-v1` policy ([results/risk_policy_fit.md](../results/risk_policy_fit.md)).

## How to open it

Open the file in a browser (double-click it, or `xdg-open docs/design/analyst-console-prototype.html`).
Click **Enter demo without credentials**. **Demo scenarios** on the Analyze page load each outcome
and error. In the yellow banner, **Prototype controls** switches theme and simulates other states,
and **Design system** opens the component reference page.

## What changed in v4 (summary; details in HANDOFF.md §10)

- The model's number is no longer on the main result. It sits in Technical details as
  "Calibrated model output", with what it is and what it is not.
- Every result follows one order: verdict, risk, required action, explanation, request
  details, technical details.
- Cancel analysis (proposal), empty-file check, skip link, focus return from dialogs, theme
  switch, Status checking and stale states, mobile filter panel, design tokens and a Design system page.

## Open decisions

Listed in HANDOFF.md §11: authentication, real upload limits, recordings longer than 2 s, fallback
retention, latency target, time zone for daily statistics, the definition of Degraded, whether the
model output appears outside Technical details, and two proposals (Cancel analysis, extra API
endpoints for Stage 17). §11 of the protocol is unchanged.

## How it was checked

49 automated browser checks and an axe-core WCAG 2.0/2.1/2.2 A/AA scan of 12 screens (light and
dark, 390 px) passed on this exact file; widths 1440, 1024, 768, 640 (200 % zoom) and 390 px.
Not yet done: manual keyboard and real screen-reader review, interpretation test with new users
(HANDOFF.md §13).

AI assistance: built and checked with Claude (claude.ai); reviewed by Iheb Zemzemi.
