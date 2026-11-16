# 0017. Analyst interface: Direction D (mix of A and B) as the D55 design

- **Status:** Proposed. Team choice 2026-11-16. Awaiting PR approval.
- **Date:** 2026-11-16
- **Register:** D65 (refines D55 (1))
- **Author:** Iheb Zemzemi
- **Deciders:** Donia Mabrouk, Malak Ben Salem, Khalil Dridi (approvers); Iheb Zemzemi (author)

## 1. Context

- The course professor judged the interface generic. D55 (`README.md` l.62, approved in PR #111)
  answers that with a redesign built around the analyst's decision, plus a usability test. D55 (1):
  three distinct directions are produced as mockups, and the team chooses one in one meeting.
- Card P5 (`docs/reference/Execution_Plan_v3.0.md:356-362`) planned the vote for Fri 13 Nov.
  The team chose on Mon 2026-11-16.
- D55 (2) requires a decision board (calibrated `p` on a 0–1 scale, the frozen threshold and band,
  and a Why panel), a per-window timeline for uploads and live sessions (D52) and the live screen (D51).
- The current design source is prototype v4 with [HANDOFF.md](../design/HANDOFF.md). Code in `web/`
  cites HANDOFF.md by line number (about 74 citations), so this decision does not move its lines.

## 2. Options

- **A, Ruler:** a stretched (logit) score ruler with the threshold and band drawn, window cells, a
  large verdict and a full "Why this verdict" explanation.
- **B, Case file:** a left rail with tabs on phones, cards on a tinted background, the decision on the left and
  evidence on the right, a tickable next-steps checklist and a History register with chips.
- **C, Triage:** the third direction; not chosen, and no element of it is taken into Direction D.

The A, B and C mockups are not committed. The chosen result is committed as Direction D (§3).

## 3. Decision

The team chose **a mix of A and B with three changes (§4)**, built as **Direction D**:
[`docs/design/prototype-d55-direction-d.html`](../design/prototype-d55-direction-d.html), sha256
`7cb2e4ae1d23d5309c7f11bf4ea6b6e17fc61d027965b14d91bc568cdc5ef34c`, specified in
[HANDOFF-v2.md](../design/HANDOFF-v2.md). This refines D55 (1): the choice is this mix, not one of
the three directions as drawn. All other D55 clauses are unchanged (§8).

| Element | From | Why |
|---|---|---|
| Shell | B: left rail on desktop, tabs under 900 px | Fits the reference look and already works at 390 px |
| Surfaces | B: cards on a tinted background | Groups content |
| Result layout | B: decision on the left (sticky), evidence on the right | The analyst reads the verdict and next steps first |
| Verdict and tiles | A's large verdict inside B's tiles | Clear verdict, compact facts |
| Score scale | A: stretched (logit) ruler | Makes the synthetic zone (above 0.997) visible on one bar |
| Windows | A: cells plus a screen-reader table | One score per 2 s window, in order |
| What to do next | B: tickable checklist, not saved (change 3) | Tracks steps on screen |
| Explanation | A: "Why this verdict" with the cut and band explained | More complete than B's |
| History | B: register inside a card, with chips | Readable list, chips carry icon and label |
| Accent | B blue #2B4C8C / #9DB8F0 | Never confused with verdict colours |
| Fonts | Public Sans + IBM Plex Mono (numbers and IDs only, change 2) | One family; 4 weights instead of today's 6 (Plex Sans 400/500/600/700 + Plex Mono 400/500) |

## 4. The three changes and where they appear in the prototype

1. **The three policy numbers are major labels, shown at every width.** 0.27, 0.633 and 0.997 are
   bold tick labels (l.491, l.124). The prototype labels this change in a comment (l.487). At ≤ 760 px the
   minor labels are hidden and 0.27 and 0.997 stay. The 0.633 tick label is hidden with the cut label (class `cut`,
   l.502, l.267) and is given in the phone legend instead (l.506). HANDOFF-v2 §5.7.
2. **One body family, with mono only for numbers and IDs.** Public Sans for text and IBM Plex
   Mono for `p`, confidence, window scores, request IDs and versions (l.15–16, l.38, l.123, l.142,
   l.149, l.171). HANDOFF-v2 §4.
3. **The "What to do next" checklist is tickable but not saved.** Checkboxes per step, with the
   note "Ticks are for your own tracking on this screen; they are not saved." (l.105–109,
   l.528–530). HANDOFF-v2 §5.6.

## 5. Colour finding and rule

As chart fills, the red and amber verdict colours fail colour-blind separation: deuteranopia ΔE 3.2
in light; for normal vision ΔE is about 11, below 15. [Team measurement for D55; not re-measured
in this PR.]

**Rule:** a verdict colour is never the only signal. Every verdict colour comes with an icon and a
label, and no chart encodes verdicts by fill colour alone. In Direction D the ruler's uncertain zone
is also hatched with a dashed edge (l.119), window cells carry icon + word (l.513), and Overview bars
use the accent colour only, with the verdict split in tooltips and tables (l.194, l.703).

## 6. Overview (S9)

The prototype includes an Overview with invented data (HTML l.302–324, script l.586–720) and its
own notice (l.304). **It is not built in D55.** The real page needs a summary endpoint, because
History returns one page at a time, and that is an API contract change. S9 stays a stub (open
item, project log). HANDOFF-v2 §6 keeps the chart specs for that later work.

## 7. Font change and why

- **Today:** IBM Plex Sans 400/500/600/700 and IBM Plex Mono 400/500, 6 weights through `@fontsource`
  (`web/src/main.tsx:2-7`).
- **Direction D:** Public Sans 400/600 and IBM Plex Mono 400/600, 4 weights, all under SIL OFL 1.1.
- **Why:** one family plus mono, fewer files to ship, and close to the reference look.
- **How:** still self-hosted. `@fontsource/public-sans` is a new dependency in B1, with `npm audit` in its
  checks. The CSP needs no change: `deploy/nginx_security_headers.conf:11` sets `default-src 'self'`
  with no font-src or style-src. No `data:` fonts and no `style=""` in served HTML; run-time positions
  go through React style props (CSSOM).

## 8. What does not change

- The Protocol §11 response fields and the API contract. Today's response is
  `web/src/api/types.ts:18-27`, exactly 8 fields.
- NFR-05 wording.
- The axe e2e scans (WCAG 2.0, 2.1 and 2.2 A and AA, light and dark) and the 390 px layout checks
  stay green.
- Colours only in `web/src/styles/tokens.css`, with `tokens.test.ts` and `colours.test.ts` updated.
- Showing confidence on the main result means updating HANDOFF.md in the same PR.
- The live screen follows D51 (and D52). Where the prototype's live screen differs (median instead of
  mean, unscored warm-up windows, no "no speech" cell, consent text), D51 and D52 win
  (HANDOFF-v2 §5.10).
- D55 (2)–(4), including the usability test (3–5 participants, SUS ≥ 68).
- **D63 dependency:** the ruler pin, `p` in the explanation and the window cells need `p`, the threshold,
  the band and per-window values, which only D63 (`README.md` l.69, Proposed) would add.
  This decision does not change the contract.

## 9. Consequences

**Easier:** one approved target with exact tokens and line-cited specs; reviewers compare each
build PR against a fixed, hashed file.

**Harder or committed:**
- **Score not available state.** Specified in HANDOFF-v2 §5.9; it is not in the prototype. It is used when
  the response has no `p` or per-window values: before D63, if D63 is rejected, and on the mock.
  - The ruler shows its three zones, with the verdict's zone outlined, no pin, and the text "Exact score not shown".
  - The "Each 2-second window" card is hidden.
  - "Why this verdict" uses band wording without `p` and no "leans …" phrase.
  - Components take `p` and `windows` as optional props, so B3 does not wait for D63.
- **Build PRs:**
  - **B1 tokens and fonts:** `tokens.css` for both themes from HANDOFF-v2 §3, `@fontsource/public-sans` and
    Plex Mono 400/600 (`npm audit`), `tokens.test.ts`, `colours.test.ts` and the contrast test,
    including the open `--line` border point (HANDOFF-v2 §3). Proof: the token table equals HANDOFF-v2 §3.
  - **B2 shell and base components:** rail and phone tabs, card, chip, buttons.
  - **B3 Analyse and Result × 3:** tiles, checklist, Why panel, ruler, window cells and their table,
    with `p` and `windows` as optional props and the Score not available state. It updates HANDOFF.md in the
    same PR (confidence on the main result). Real values only after D63.
  - **B4 Live check in Direction D, delivered inside L2** (`Execution_Plan_v3.0.md:315-319`,
    D51/D52). It is not a separate D55 build PR.
  - **B5 History and the empty state.**
  - Whichever of B3, B5 and the L2 live screen merges last retires HANDOFF.md as the description of the
    shipped UI. HANDOFF-v2 §9 says the same.
- **What every build PR shows** (HANDOFF-v2 §10):
  - side-by-side screenshots of the build and the prototype at 1440, 1024 and 390 px, in light and dark;
  - the HANDOFF-v2 sections covered, ticked, with any difference named;
  - axe e2e and 390 px checks green;
  - no raw colour outside `tokens.css`.
  B3 also shows the Score not available state.

**Risks:** the build may drift from the prototype, which the screenshot comparison catches. If D63 is
rejected, the ruler stays without a pin, and D55 (2)'s "calibrated p on a 0–1 scale" is then met only partly.
That would need a team decision.

## 10. References

- D55, D51, D52, D63, D38 / [ADR 0008](0008-web-frontend-react.md), [ADR 0014](0014-live-analysis.md),
  [ADR 0016](0016-calibrated-spoof-probability.md)
- [HANDOFF-v2.md](../design/HANDOFF-v2.md), [HANDOFF.md](../design/HANDOFF.md),
  [prototype](../design/prototype-d55-direction-d.html)
- Execution Plan v3.0 card P5 and L2; project log Entry 191
