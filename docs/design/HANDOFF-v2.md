# Analyst console: design handoff v2 (Direction D)

**Target design. Implemented by the D55 build PRs.**

Design reference for the D55 redesign (decision D65, [ADR 0017](../decisions/0017-ui-direction-d.md)).
Until the build PRs replace it, [HANDOFF.md](HANDOFF.md) still describes the shipped UI (§9).
"l.N" means line N of the prototype file. Where this document and the prototype differ, this
document wins, and the difference is named.

Contents: 0 Source and status · 1 Screens and states · 2 Layout and breakpoints · 3 Tokens ·
4 Fonts · 5 Components · 6 Overview charts · 7 Content rules · 8 Accessibility ·
9 What HANDOFF.md still governs · 10 Build acceptance checklist

---

## 0. Source and status

- **Prototype:** [`prototype-d55-direction-d.html`](prototype-d55-direction-d.html), one offline
  HTML file with no network requests. Its fonts are embedded as `data:` URIs (l.8–11).
  sha256 `7cb2e4ae1d23d5309c7f11bf4ea6b6e17fc61d027965b14d91bc568cdc5ef34c`. Never edit it. A new
  design version is a new file with a new hash.
- **How to open:** open it in a browser. The rail switches screens. `?theme=dark` or `?theme=light`
  forces a theme (l.458). `?res=synthetic|uncertain|authentic` picks the result and `?empty=1` empties
  History (l.721–722). The demo rows on each screen (l.343, l.348, l.373) are prototype controls,
  not product UI.
- **Data:** every value is invented, as the prototype says at l.380, l.585 and l.595. The Overview carries
  its own notice (l.304). The policy numbers (l.381) are `risk-v1`'s, rounded:
  threshold 0.633 (`docs/results/risk_policy_fit.md:58`, 0.6333…) and band 0.270 to 0.997 (±0.3636…,
  `docs/results/stage11_for_test.md:81`).
- **Origin:** B's frame (rail, cards, checklist) with A's instruments (ruler, window cells, verdict,
  explanation) (l.12–13). The element-by-element mapping is in ADR 0017 §3.

## 1. Screens and states

| Direction D screen | Replaces (HANDOFF.md §1) | States in the prototype | Lines |
|---|---|---|---|
| Overview | S9 | loaded (invented data) | HTML l.302–324, script l.586–720 |
| Analyse | S2–S5 | idle; unsupported format; progress | l.326–344 |
| Result | S6 | authentic / LOW, uncertain / MEDIUM, synthetic / HIGH | l.346–350, l.392–417, l.516–547 |
| Result, score not available | S6 | **not in the prototype**, specified in §5.9 | — |
| Live check | new (D51, D52; built in L2) | idle, consent not given, listening, stopped | l.352–369, l.558–574 |
| History | S7 | results | l.371–375, l.549–555 |
| History, empty | S7 "no analyses yet" | empty | l.551 |

- **Overview (S9): designed, not built in D55.** It needs a summary endpoint, because History returns
  one page at a time (l.304). That is an API contract change, so S9 stays the current stub
  (ADR 0017 §6).
- **Analyse:** the prototype shows only representative states. The build keeps every S2–S5 state
  in HANDOFF.md §1 (limits, file checks, cancel, error kinds) and draws them with the new
  components. The Requirements panel (l.335–339) shows the limits from `GET /api/limits` as it does
  today, not the prototype's literal values.
- **Result:** the same layout for all three verdicts. Only the verdict, tiles, checklist steps,
  explanation and window cells change (l.392–417).
- **S1 Sign in, S8 Analysis details and S10 System status** are not in the prototype. They keep
  their HANDOFF.md layout and take only the new tokens (§3) and components (§5). S8 stays a stub,
  as it is today.
- **Live check** follows D51 and D52. Where the prototype differs, the decisions win (§5.10).

## 2. Layout and breakpoints

| Width | Navigation | Layout | Lines |
|---|---|---|---|
| > 1100 px | Left rail 232 px, sticky, full height | Content padding 32 px, max width 1240 px. Result: decision card left (5 fr, min 300 px, sticky at 24 px), evidence right (7 fr). Overview: 4 KPIs, charts 7 : 5 | l.44–45, l.63, l.93–94, l.180, l.187 |
| ≤ 1100 px | Rail | Overview: 2 KPIs per row, charts stack | l.245 |
| ≤ 900 px | The rail becomes a top bar; the nav becomes a horizontal tab row (order 3, scrolls if needed); the user box is hidden | Single column; the decision card is no longer sticky; content padding 16 px | l.247–258 |
| ≤ 760 px | — | Verdict 32 px; card padding 16 px; ruler 98 px high with minor tick labels, zone labels and the cut label hidden, and the phone legend shown; window cells become one row per window with a 5 px left border; tiles stack; the History table becomes stacked rows with labels; fact lists stack | l.263–280 |
| ≤ 600 px | 5 equal tabs, icon above label, 12 px text, 52 px high | Overview: the ring stacks above the chart; KPIs 2 per row | l.244, l.246, l.259–262 |

Kept from HANDOFF.md §3: no horizontal page scroll; never truncate verdicts, risk labels, request IDs
or dates (IDs wrap, l.96, l.171); every target ≥ 44 × 44 px (l.51, l.58, l.60, l.106). The checks at
390 px stay green.

## 3. Tokens

All colours live in `web/src/styles/tokens.css` (l.13), and `tokens.test.ts` and `colours.test.ts` are
updated in the same PR. The table copies l.14–31 exactly.

### Colour

| Token | Light (`:root`, l.20–24) | Dark (`[data-theme="dark"]`, l.27–31) | Use |
|---|---|---|---|
| `--bg` | `#F4F6FA` | `#0F1520` | page background (tinted) |
| `--surface` | `#FFFFFF` | `#18202D` | cards, rail |
| `--surface-2` | `#F7F9FC` | `#1D2735` | tiles, table heads, hover |
| `--ink` | `#172033` | `#E4E9F1` | text |
| `--ink-2` | `#4D5A6E` | `#A9B4C5` | secondary text |
| `--line` | `#E3E8EF` | `#2A3546` | dividers, card borders (decorative only, see below) |
| `--rule` | `#5E6B76` | `#8796A3` | ruler axis and ticks |
| `--accent` | `#2B4C8C` | `#9DB8F0` | primary button, current tab, chart bars and lines |
| `--accent-soft` | `#E8EEF9` | `#22304A` | current tab background, notices |
| `--accent-ink` | `#FFFFFF` | `#0F1520` | text on `--accent` |
| `--auth` / `--auth-bg` | `#1F6E43` / `#E1F0E6` | `#6DCB97` / `#16301F` | authentic |
| `--unc` / `--unc-bg` | `#8C5300` / `#F8EAD2` | `#F0B25E` / `#34270F` | uncertain |
| `--syn` / `--syn-bg` | `#A3262A` / `#F7E1E1` | `#FF948C` / `#3A1B1C` | synthetic |
| `--focus` | `#2B4C8C` | `#9DB8F0` | focus ring |
| `--live` | `#A3262A` | `#FF948C` | listening indicator |
| `--shadow` | `0 1px 2px rgb(16 24 40 / .06), 0 1px 3px rgb(16 24 40 / .05)` | `0 1px 2px rgb(0 0 0 / .35)` | cards |

Theme: the system preference applies by default and the user can switch (l.455–463). The build keeps
its existing theme mechanism and switches tokens on `[data-theme]`.

**Measured contrast (WCAG 2.x, computed from the hex values above):** every text pair used is at least
5.2 : 1 in both themes (the lowest are verdict text on its own background: light 5.28, dark 7.21).
`--rule` on `--surface` is 5.47 / 5.39. `--line` on `--surface` is only 1.23 / 1.32, so it may be
used only where it is decorative. Its use as the border of `.quiet` buttons (l.58) and window cells
(l.139) is an **open point for B1**: either show that these boundaries are not needed to identify the
control, or add a stronger border token, as HANDOFF.md §4 did with `--line-strong`.

### Type, space, shape

| Group | Tokens (l.15–19, l.24) |
|---|---|
| Families | `--font-body` Public Sans; `--font-num` IBM Plex Mono (§4) |
| Sizes | `--fs-label 12.5px` (uppercase labels, 600, letter-spacing .06em, l.41), `--fs-xs 13px`, `--fs-sm 14px`, `--fs-md 16px` (body, line height 1.55, l.34), `--fs-lg 19px` (h2), `--fs-xl 26px` (h1), `--fs-verdict 40px` (32 px at ≤ 760 px) |
| Space | `--sp-1 4px`, `--sp-2 8px`, `--sp-3 12px`, `--sp-4 16px`, `--sp-5 24px`, `--sp-6 32px`, `--sp-7 48px` |
| Radii | `--r-sm 8px` (nav items, quiet buttons), `--r-md 10px` (buttons, tiles, cells), `--r-lg 14px` (cards), `--r-pill 999px` (chips) |
| Elevation | `--shadow` on cards only; tooltips use their own shadow (l.201) |
| Motion | new-window pop 0.4 s and listening pulse 1.2 s, only under `prefers-reduced-motion: no-preference` (l.145, l.159) |

## 4. Fonts

- **Public Sans** 400 and 600, and **IBM Plex Mono** 400 and 600, all under the SIL Open Font
  License 1.1 (l.8–11, l.15–16). That is 4 weights; today there are 6 (`web/src/main.tsx:2-7`).
- **Self-hosted** woff2 through `@fontsource/public-sans` (a new dependency in B1, with `npm audit`
  in its checks) and the existing `@fontsource/ibm-plex-mono`, served from `/assets/`.
- **Mono only for numbers and IDs:** `p`, confidence, window times and scores, request IDs, version
  strings, Technical details (l.38, l.123, l.142, l.149, l.171). Labels and prose use Public Sans.
- **CSP:** nothing to add. `deploy/nginx_security_headers.conf:11` sets `default-src 'self'` with no
  `font-src` or `style-src`, so same-origin fonts and stylesheets are allowed. The build must not use
  `data:` fonts, which that policy blocks (the prototype embeds them only to work offline), or put
  `style=""` attributes in served HTML. Positions computed at run time (ruler zones and pin, l.495–504)
  are set through React style props (CSSOM). The prototype's inline styles (e.g. l.308, l.535) become
  classes.

## 5. Components

### 5.1 Rail and phone tabs (l.43–61, l.285–299)
- Brand block, a "Workspace" label, then 5 items, each with an icon and a label: Overview, Analyse,
  Result, Live check, History. The theme button and the user box are in the footer.
- Current item: `aria-current="page"`, `--accent-soft` background, `--accent` text (l.53).
- Changing screen moves focus to that screen's `h1` (l.447).
- ≤ 900 px: tabs in a row; ≤ 600 px: 5 equal tabs (§2). The build keeps today's routes. The Result
  tab is the prototype's way to reach a result and is not a new route.

### 5.2 Card (l.70)
`--surface`, 1 px `--line` border, `--r-lg`, `--shadow`, padding 24 px (16 px at ≤ 760 px). Cards on
`--bg` group content.

### 5.3 Chip (l.75–77)
A pill with an icon and a label in the verdict colour on its `-bg` (`authentic`, `uncertain`,
`synthetic`), or neutral (`warm`). It is used for risk in the result (l.526), for verdicts in History
(l.554) and in the Overview list. A chip never shows colour without its icon and text.

### 5.4 Buttons (l.58–61)
- `.btn`: primary, `--accent` / `--accent-ink`, `--r-md`, min height 44 px, disabled at 45 % opacity.
- `.quiet`: secondary, `--surface` with a 1 px border (open point in §3), min height 44 px.

### 5.5 Tiles (l.100–104, l.524–527)
Two tiles under the verdict: **Prediction · confidence** (verdict word + confidence in mono + a
hint) and **Risk level** (a chip). They stack at ≤ 760 px. Hint text (l.525): uncertain → "How far p
is from 0.5, not how sure the verdict is."; otherwise → "Confidence is max(p, 1 − p)." Showing
confidence on the main result changes HANDOFF.md §8; B3 updates HANDOFF.md in the same PR (D55 rule).

### 5.6 "What to do next" checklist (l.105–109, l.528–530)
- Native checkboxes, one per step, each row a 44 px label.
- **Ticks are not saved** (change 3). The note under the list says so: "Ticks are for your own
  tracking on this screen; they are not saved." Nothing is sent or stored, and leaving the screen
  clears them.
- The steps per verdict are at l.399, l.407 and l.415. Final wording goes through the content review (§7).

### 5.7 The ruler (l.113–132, l.487–509)
- **Mapping:** stretched logit scale. `pos(p) = (logit(c) + m) / (2m)` with
  `c = clamp(p, 0.001, 0.999)` and `m = logit(0.999)` (l.390). Equal distances are not equal
  differences in `p`.
- **Zones:** authentic from 0 to 0.27 (solid, `--auth-bg` with a solid 3 px `--auth` underline);
  uncertain from 0.27 to 0.997 (diagonal hatch with a **dashed** `--unc` underline); synthetic above 0.997
  (solid, `--syn-bg` with a solid `--syn` underline) (l.117–120, l.495–497). Each zone has a label with an icon (l.498–500).
- **Ticks:** 0.01, 0.1, 0.27, 0.633, 0.9, 0.99, 0.997 (l.490). **Major labels 0.27, 0.633 and 0.997**
  are bold (l.491, l.124), and all three numbers are shown at every width (change 1, l.487). At ≤ 760 px the
  minor labels are hidden. The 0.27 and 0.997 tick labels stay. The 0.633 tick label carries class `cut`
  (l.502) and is hidden with the cut label (l.267), so on phones 0.633 is given in the legend
  ("┊ yes/no cut 0.633", l.506). A dotted line marks the yes/no cut at 0.633, labelled "┊ yes/no cut" (l.503).
- **Pin:** `p` in mono on an `--ink` pill above a 3 px stem; it right-aligns past 80 % of the bar
  (l.127–130, l.504).
- **Phone legend** (≤ 760 px, when the zone labels are hidden): "Authentic below 0.27", "Uncertain
  0.27 to 0.997", "Synthetic above 0.997", "┊ yes/no cut 0.633", each with its icon (l.269–270, l.506).
- **Scale note** under the bar, always shown: "The scale is stretched near 0 and 1 so the narrow
  synthetic zone (above 0.997) is visible. Equal distances are not equal differences in p." (l.507)
- **Screen readers:** the drawing is `aria-hidden`. A visually hidden caption reads `p`, the three zones,
  the cut and the zone of this clip (l.480–482, l.493).
- **Numbers come from the API, never from the browser.** The prototype computes the verdict from `p`
  with `p > 0.997` → synthetic (l.385). The frozen policy treats `p ≥ 0.997…` as synthetic (band
  `[t − b, t + b)`, `stage11_for_test.md:81`). The build draws zones from the response's threshold
  and band (D63) and takes the verdict from `prediction`. It never derives a label from `p`.

### 5.8 Window cells and their table (l.137–145, l.511–514, l.539–541)
- One cell per 2 s window, in order: the time range, the verdict with its icon, and `p` in mono. A
  5 px top border in the verdict colour (a left border at ≤ 760 px). Grid of cells at least 112 px wide; one per row at ≤ 760 px.
- States: scored (authentic / uncertain / synthetic); warming up (dashed border, warm icon, "not scored",
  l.141, l.513); new in live (accent outline, l.144).
- The cells are `aria-hidden`. A visually hidden table carries the same rows (Window, p, Verdict)
  (l.541). In live, a visually hidden list gets one item per window (l.366, l.567).
- Window labels come from the API (D52 (4)), not from `p` in the browser (§5.7).

### 5.9 Score not available (not in the prototype; specified here)
Used when the response has no `p` or no per-window values: before D63 is approved, if D63 is rejected,
and on the mock analyzer.
- **Ruler:** the three zones and the tick labels as in §5.7. The zone of the verdict in `prediction`
  is outlined (2 px `--ink` outline, not a fill change). No pin. The text "Exact score not shown" is
  in place of the pin, and the hidden caption reads "Exact score not shown. This clip is in the
  <verdict> zone.".
- **"Each 2-second window" card** (l.539): hidden, along with its table.
- **"Why this verdict":** band wording without `p`, e.g. "This result is inside the uncertain band
  (0.270 to 0.997), where the model was wrong too often on validation data, so it defers the
  decision to you." No "Leans …" phrase (l.397, l.523) and no "p = …".
- **Tiles, checklist, notice and Technical details** are unchanged. Confidence stays in the tile.
- **Build rule:** `p` and `windows` are optional props of the ruler, result and window components.
  The component shows this state when they are absent, so B3 does not wait for D63.

### 5.10 Live check (l.151–161, l.352–369, l.465–478, l.557–574; built in L2)
- **Decision card:** "Before you start" text, a decision-support notice, the consent checkbox ("I have told the
  caller."), then Start (disabled until the box is ticked) and Stop. "Listening to your microphone" uses a `--live` dot
  (`role="status"`). The stopped state reads "Stopped. Nothing was recorded." The running verdict uses a polite live region.
- **Evidence:** "Running score" ruler and "Windows so far" cells.
- **D51 and D52 override the prototype here:**
  - The running verdict is the frozen threshold and band applied to the **mean** calibrated `p` of all
    kept windows since Start (D52 (5)). The prototype uses the median (l.365, l.466, l.473).
  - Warm-up means **no running verdict before 3 kept windows** (D52 (6)). The windows are still
    scored and count toward the mean. The prototype instead leaves the first 3 windows unscored and outside the
    running value (l.366, l.471–472).
  - A window that fails the speech filter is shown as **no speech** and does not enter the verdict
    (D52 (3)). The prototype has no such cell; the build adds one (neutral, dashed, icon + "No speech").
  - The mean of the last 5 kept windows is shown as a trend line only, without a label (D52 (7)).
  - The UI states that the other party's consent may be legally required (D51). The prototype's text
    (l.357) is replaced by that statement.

### 5.11 Tooltip (l.201–206, l.645–654, l.697–712, l.719)
- A card-like box with a value, a date line and a list of label/value rows (mono values).
- Opens on pointer hover **and on keyboard focus** of the same data point, closes on leave, blur or
  Escape.
- Its content is built as text nodes, not HTML strings (l.646–651).

### 5.12 Table view (l.207–209, l.308, l.655, l.674, l.714–717)
- Every chart has a "Show as table" button (`aria-expanded`, `aria-controls`) that shows a data
  table with the chart's rows plus a Change column. The label switches to "Hide table".

### 5.13 Other parts of the Result screen
- **Case meta:** the file name, duration and time (l.95–96, l.520).
- **Verdict:** a label, the verdict word in `--fs-verdict` with its icon, and the long form (l.97–99, l.430–431, l.521–523).
- **Decision-support notice** (l.110, l.531).
- **"Where this clip falls"** card: the ruler, "Why this verdict" and the facts list, which gives the yes/no cut as the
  equal-error-rate point on val-B, the band as sized so at most 10 % of validation clips are deferred, and the model version
  (l.534–538).
- **Technical details:** collapsed; lists all 8 §11 fields in mono (l.542–545).

### 5.14 History register (l.163–173, l.549–555)
- The table sits inside a card. Columns: Analysed (with the time zone), Verdict (chip), Risk, Request ID
  (mono, wraps) and Analyst note. At ≤ 760 px each row becomes a stacked block with labels (l.275–278).
- The build keeps today's filters, paging and time-zone rule from HANDOFF.md S7.
- **Empty state:** "No analyses yet", "Results appear here after your first analysis.", and an
  "Analyse a recording" button (l.551).

## 6. Overview charts (designed, not built in D55; HTML l.302–324, script l.586–720)

For the later S9 work. All data in the prototype is invented (l.304, l.585, l.595).

- **Layout:** 4 KPI cards (Analyses in 7 days, Synthetic, Uncertain, Authentic). Each has an icon, a
  label, a count, a share and the previous 7 days. Then "Analyses per day", "Uncertain rate" and "Needs your
  attention" (with "Verdict split, last 7 days") (l.180–186, l.305–322, l.659–672).
- **Analyses per day:** bars in **`--accent` only**; a hovered or focused bar turns `--ink` (l.194,
  l.197). The verdict split per day is in the tooltip and the table, never in bar colours. Values are
  labelled on the highest and the last bar. Every other x-label is shown under 520 px (l.676–694).
- **Uncertain-rate ring:** 0–100 % with an `--accent` arc and a 10 % target tick. The text beside it gives
  "u of n analyses, 7 days to <date>" and "Target 10% (tick) · above / at or below target" (l.232–242,
  l.614–625). Its label reads the value and the target.
- **7-day rolling uncertain line:** an `--accent` line with dots; a dashed "Validation target 10%"
  line; the last point is labelled with its value (l.226–230, l.609–613, l.626–643).
- **Version-change markers:** dotted vertical lines on the first day a new model or policy version
  was used, with full labels, or short ones ("Model", "Policy") under 520 px. Labels move to a second row if
  they would overlap (l.224–225, l.595–608).
- **Verdict split bars:** neutral `--ink-2` fill; verdicts are carried by the icon and the label
  (l.218–222, l.672).
- **Every chart** has a table view (§5.12) and the same tooltip on keyboard focus as on hover
  (§5.11). Each data point is a focusable `role="img"` with a full text label (l.639, l.690).
- **No chart encodes verdicts by fill colour** (the colour finding, ADR 0017 §5).
- **Data the endpoint must give** (open item): per-day counts per verdict, rolling uncertain counts,
  version-change days. Whether `p` and the clip duration are stored for History is still to be checked.

## 7. Content rules

- NFR-05 wording is unchanged: "Decision support only. This result is not proof of fraud." (l.383).
  The live variant reads "A live result is not proof of fraud." (l.358).
- Verdict words and long forms as at l.430–431. The words HANDOFF.md §8 avoids stay avoided.
- `p` is written "p" with 3 decimals, or 4 when p ≥ 0.995 (l.388). Confidence has 2 decimals, or 3 when ≥ 0.995
  (l.389).
- The "Why this verdict" text names the band and the cut and how they were derived (val-B EER point,
  10 % deferral, model version) (l.398, l.406, l.414, l.536–538). It has no `p` in the Score not available state (§5.9).
- The checklist steps (l.399, l.407, l.415) are a proposal. The final wording is checked in the D55 usability test.

## 8. Accessibility

- **Every verdict colour comes with an icon and a label:** the verdict word, chips, zone labels, legend,
  cells and tooltips (icons l.433–440).
- **No chart encodes verdicts by fill colour alone.** The ruler's uncertain zone also differs by hatch
  and a dashed underline (l.119). Cells carry icon + word. Overview bars are accent only.
- **Every chart has a table view**, plus keyboard focus on each data point with the same tooltip
  (§5.11, §5.12). Escape closes the tooltip.
- The focus ring is 3 px `--focus` with a 2 px offset (l.35). Targets are at least 44 × 44 px. One `h1` per screen, and focus moves
  to it on screen change (l.347, l.447). Live updates use `role="status"` / `aria-live="polite"`.
- Gates (D55, unchanged): axe e2e with WCAG 2.0, 2.1 and 2.2 A and AA tags in light and dark, and the
  390 px layout checks, stay green.

## 9. What HANDOFF.md still governs

- Until each screen is rebuilt, HANDOFF.md describes the shipped UI. It also stays the source for
  what this document does not cover: S1, S8, S10, dialogs and toasts, the error kinds and their wording
  (§8), and the developer notes (§12). Code comments citing HANDOFF.md by line stay valid because
  its line numbers do not change in this PR.
- Whichever of B3, B5 and the L2 live screen merges last retires HANDOFF.md as the description
  of the shipped UI.
- **D63 dependency.** Today's response (`web/src/api/types.ts:18-27`) has exactly the 8 §11 fields
  and no `p`, threshold, band or per-window values. D63 (`docs/decisions/README.md:69`, Proposed)
  would add them. Until it is approved and implemented, results use §5.9.

## 10. Build acceptance checklist

Each D55 build PR (ADR 0017 §9) shows:
- [ ] side-by-side screenshots of the build and the prototype at 1440, 1024 and 390 px, in light and dark;
- [ ] the sections of this document it covers, ticked, with any difference named;
- [ ] axe e2e and the 390 px checks green;
- [ ] no raw colour outside `tokens.css` (the `colours.test.ts` check), and no `style=""` in served HTML;
- [ ] B3 only: the Score not available state (§5.9), in screenshots and in a test; HANDOFF.md updated
  for confidence on the main result.
