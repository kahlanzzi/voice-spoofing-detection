# Design: analyst console prototype

`analyst-console-prototype.html` is the clickable prototype of the analyst interface approved in
D38 / [ADR 0008](../decisions/0008-web-frontend-react.md). It is a **design reference for Stage 18,
not production code**: one self-contained HTML file with mock data, no backend and no model.

Version 3 (2026-10-29) answers the second UI/UX review (navigation, result wording, upload and
error states, History filters, Overview labels, responsive layout, accessibility, sign-in states,
System status). It also follows the fitted `risk-v1` policy
([results/risk_policy_fit.md](../results/risk_policy_fit.md)).

## How to open it

Open the file in a browser (double-click it, or `xdg-open docs/design/analyst-console-prototype.html`).
Click **Enter demo without credentials**. **Demo scenarios** on the Analyze page load each outcome
and error. The yellow banner's **Prototype controls** switch on other states: service or API down,
upload limits unavailable, expired session, each sign-in failure, note save failure, History and
Overview errors.

## What it covers

- The four areas of D38: Analyze (default after sign-in), History, Overview, System status, plus
  sign-in and a separate Analysis details view with **Back to History** (filters, page and scroll
  are kept; browser Back and Forward work the same way).
- Three verdicts in words, colour and icon: **Likely synthetic voice** (HIGH), **Cannot decide**
  (MEDIUM), **Likely authentic voice** (LOW), with explanation, next steps, request ID, processing
  details and an optional analyst note that never changes the result.
- Upload: accepted formats, maximum size, maximum duration and the deletion rule are read from a
  mock `GET /limits` response, never hard-coded. If they cannot be loaded, analysis stays off.
- Errors, none of which shows a verdict or score: unsupported format, file too large, recording too
  long, unreadable audio, service unavailable, request timeout, network interruption, expired
  session, unexpected server error. Each has its own action (Retry, Choose another file, Sign in
  again).
- History: multi-select verdict filter, date range, request-ID search, result count, Clear filters,
  pagination, loading, no-results and failed-to-load states. No audio file names.
- Overview (new scope, D38 item 8): operational figures only, each with date range, time zone and
  last-updated time, a table view for each chart, and loading, unavailable, stale and empty states.
  No accuracy, EER, per-analyst figures or DEEP-VOICE data.
- System status: a summary first (Ready for analysis / Degraded / Unavailable), technical details
  (bundle verification, versions) folded away.
- Uses only the frozen §11 response fields. §11 is unchanged.

## Findings for team review

1. **The `confidence` field with `risk-v1`.** `confidence = max(p, 1 − p)` is the calibrated
   probability of the more likely class, not of the verdict. With the fitted band
   (`p` 0.270–0.997), every synthetic result shows **1.00**, and a Cannot decide result can show any
   value from 0.50 up to 1.00 ([risk_policy.md](../risk_policy.md) Why (d); fit §3 item 4). The
   prototype shows it as **Model score**, always next to the verdict, with that explanation. The
   label, and whether the score belongs on the main result at all, is for the team to decide.
   The API field keeps its name.
2. **Wrong README on `main`.** The README committed with the first version (PR for Entry 84) was an
   unrelated file. This README replaces it.

## Open decisions shown as placeholders

Access-control method (SRS §7.4; the sign-in screen shows the analyst-login option only, without
password recovery or a sign-out confirmation); the real upload limits; handling of recordings longer
than 2 s; the fallback audio-retention limit and the latency target (removed from the screens until
decided); the time zone used to count days in statistics (shown as UTC); what counts as Degraded.

## API needs beyond §11 (for Stage 17 design, team review)

Upload limits (`GET /limits`), statistics (`GET /stats`), service status, paginated and filtered
history (verdict, date range, request ID), analyst notes, and distinct error codes for unreadable
audio, timeout, expired session and server error. §11 itself is unchanged.

## How it was checked

46 automated browser checks (Playwright, Chromium, time zone Africa/Tunis) passed on this exact
file, with no console errors. They cover every sign-in state, upload limits and their failure,
all input and service errors, the three verdicts, a single request under repeated clicks and
navigation, notes, History filters and return, Overview and Status states, page-change focus,
scroll and announcement, reduced motion, dark mode, and widths 1440, 1024, 768 and 390 px (no
sideways scrolling, 44 px touch targets at 390 px, History as cards). An axe-core 4.10 scan
(WCAG 2.0/2.1/2.2 A and AA rules) found no violations on 10 screens, including dark mode and
390 px. Not checked: real mouse drag-and-drop and real screen readers.

AI assistance: built and checked with Claude (claude.ai); reviewed by Iheb Zemzemi.
