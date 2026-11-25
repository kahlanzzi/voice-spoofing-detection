# VoiceCheck Enterprise: proposed interface direction

**Status: proposal for Iheb and team review, not the approved D55/D65 target.** The current target
remains [Direction D](prototype-d55-direction-d.html) and [HANDOFF-v2](HANDOFF-v2.md). This page
does not change a decision, API contract, model, policy, or the running React application.

## Open the candidate

Download or open [the standalone HTML prototype](voicecheck-enterprise-proposal.html) in a browser.
It is Malak's supplied `VoiceCheck_Enterprise.html`, copied without edits (SHA-256
`390f35c885cf6d84d80fc663d7fb8dfe8a95c689760e9b09a28a6fd8fd5dacd6`). It simulates
sign-in, upload, live analysis, reviews and administration in the browser. It does not call the API,
capture audio, create accounts or persist records. The Inter font is loaded from an external URL,
so the appearance may differ without an internet connection.

## What Iheb should review

| Area | Candidate | Approved direction or contract | Required follow-up if adopted |
| --- | --- | --- | --- |
| Layout | Adds an enterprise overview, review queue and administration screens to the analyst workflow. | D65 Direction D is the target. Its Overview is a stub until a summary API is approved. | Decide which new screens belong in D55 and specify their API, access and storage behavior separately. |
| Decision board | Shows a nonlinear score scale and policy details, but its main scale omits the 0.633 cut label and does not make all three D65 policy numbers major labels. | D55/D65 require the calibrated probability, threshold and band on the decision board; 0.270, 0.633 and 0.997 are prominent across widths. | Bring the board into conformance or approve a specific amendment to D65. |
| Typography | Uses Inter for body text. | D65 specifies Public Sans and IBM Plex Mono for numbers and IDs. | Choose the final font through a reviewed D65 amendment if it changes. |
| Live demo | Uses the median of later windows and describes the first three windows as unscored. | D52 uses the **mean of every kept window**; the first three kept windows contribute to the mean, but the verdict is withheld during warm-up. | Correct the prototype and implementation to D52 before any live build. |
| Demo data | Shows invented filenames, local verdicts, review notes and member changes. | D63 values come from the API; FR-09/NFR-04 prohibit storing original audio filenames. | Keep these as clearly labeled demo controls. Define server behavior and privacy handling before implementation. |

The current [API design](../api_design.md), D51/D52 live decisions, D55/D63 application contract,
and D65 UI decision control the implementation. The candidate is a way to discuss a possible new
visual and workflow direction with Iheb, not permission to replace them. Iheb should identify
backend implications and sequencing alongside his active Stage 17 P1 integration PR before the
team decides whether to amend D65. If the team adopts it, make a new reviewed handoff and tests;
preserve the existing hashed Direction D prototype as history.

## Verification of this proposal

The supplied HTML was copied byte for byte and its inline JavaScript passed `node --check`.
No browser interaction, accessibility scan, 390 px check or usability test has yet been recorded
for this candidate. Those checks remain required by D55 before implementation is accepted.
