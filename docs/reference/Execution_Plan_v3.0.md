# Voice-Spoofing Detection for Fraud Prevention

## Execution Plan v3.0

*Execution Plan v3.0: from the midterm to delivery, with model v2, live analysis and production, stage by stage, and who owns what*

- **Team:** Donia Mabrouk · Iheb Zemzemi · Malak Ben Salem · Khalil Dridi
- **Version / date:** v3.0 · 10 November 2026 (replaces v2.1 and the unadopted v2.2 draft) · **status updated 26 November 2026** (`main` `15cb08d`)
- **PDF:** `Execution_Plan_v3.0.pdf` is the 10 November baseline; this Markdown is the current plan. The PDF is regenerated once, at the feature freeze (Fri 18 Dec, D69 (4)).
- **Repository:** github.com/kahlanzzi/voice-spoofing-detection
- **Controlling documents:** Decision register (D1–D70, D66 reserved) › SRS v4.1 › SDD v2.0 (draft for team review) › Protocol v1.0 + addendum › this plan › Proposal

### Where we are today (26 November 2026)

| | Item | Detail |
|---|---|---|
| **DONE** | Stages 0–12 | Stage 11 official run once (Entry 161) and closed (Entry 163); F4 `bundle-v1` frozen 12 Nov (`edf4a205…40efc`, D60, Entry 168), hand-over verified by Iheb and Khalil 15 Nov (Entry 174). |
| **DONE** | Track V code and rehearsal | V1 corpora, augmentation, S and the training resource cache (evidence #163, Entry 215); V2 candidates and training code; V4–V7 code and notebooks 09–10 (PRs #129–#159). T4 rehearsal by Khalil 23 Nov (`rehearsal/20261123T220530Z/`): all five checks passed, cut line 29.5 h (23.0 h without `aasist`) against 95 h; `aasist` kept on timing (D48 (6)), decided before any search result [stated by Donia]. |
| **DONE** | P1 and P2 | P2a PostgreSQL (#155, Entry 207); P2b process pool (#162, Entry 214); P1 F4 bundle in the process pool (#165, Entry 217). |
| **DONE** | Decisions | D63 (calibrated p) and D65 (UI Direction D) approved 17 Nov; SRS v4.1 adopted (Entry 206); D67 (VoiceCheck Enterprise, replaces Direction D) and D68 (identity and roles) approved 25 Nov (#167, #168, Entries 219–220); D69 (delivery scope) and D70 (staging removed; the professor's request withdrawn in writing, 25 Nov) approved 26 Nov (#170, #171, Entries 222–223). |
| **NOW** | Track V search (V3) | Running by relay from `d5ffe18`, fixed until the search ends: session 1 Tue 24 Nov (`cnn-aug` epochs 1–16), session 2 Wed 25 Nov (epochs 17–25, resumed after epoch 16) [stated by Donia]. Selection target Fri 27 Nov, may slip to Sat 28 Nov (estimate of 25 Nov). |
| **NOW** | Stage 14 v1 | Follow-up merged (#164, Entry 216); **official run waits** for review and the all-four H6 approval. |
| **NOW** | D55 build (P5) | Phase 1, by Thu 10 Dec: prototype corrections, HANDOFF-v3, B1 + shell, note filter, Analyse, Result, History, Live check (D67; prototype #166, Entry 218). |
| **LATE** | Track P hosting (P3) | Iheb's D54 (9) hosting amendment is in progress (not merged): one free VM (D70 (5)); Vercel cannot run the stack [stated by Iheb, 26 Nov]. The production host follows it (no staging, D70), then P4 and production acceptance. |
| **DATE** | Hard dates | v2 freeze (F5) cut-off: Fri 4 Dec 2026, 23:59 Tunis (D48 (4)). Feature freeze: Fri 18 Dec 2026 (D69 (4)). Delivery target: Wed 23 Dec 2026, to be confirmed against the course calendar. |

## Contents

- 1\. How to use this plan
- 2\. What changed after the midterm
- 3\. The project on one page
- 4\. Hard rules
- 5\. How we work
- 6\. Team, ownership and reviews
- 7\. Timeline and critical path
- 8\. Track R — v1 to its freeze (Stages 11–12)
- 9\. Track V — model v2 (V1–V5)
- 10\. Part D — joint external run and analyses (13–16)
- 11\. Track L — live analysis (L1–L3)
- 12\. Track P — platform and production (P1–P5, 19)
- 13\. Risks and mitigations
- 14\. Open items
- Appendix A — Decisions since v2.1
- Appendix B — Freeze points and registry claims
- Appendix C — Document versions
- Appendix D — What changed since v2.1

## 1. How to use this plan

This is the team's working reference from the midterm to delivery. It says what we build, in which order, by when, and what "done" means at each step. It does not repeat the rules in full: the decision register, the protocol and `AGENTS.md` hold them, and this plan points to them.

- **Start of a step:** read its card (Sections 8–12): goal, tasks, outputs, gate, dates and the decisions that apply.
- **During a step:** follow the hard rules (Section 4) and the workflow (Section 5).
- **End of a step:** done only when its gate is passed: outputs merged into `main` through a reviewed PR with green CI, and the project log updated.
- **Late or blocked?** Say it in the weekly sync the same day. The critical path (Section 7) shows which delays move the 4 December cut-off and which do not.
- **Something conflicts?** The decision register wins, then SRS, then SDD, then the protocol, then this plan. Never resolve a conflict silently.

> **Why v3.0?.** v2.1 planned one CNN, an upload-only service and a DEEP-VOICE run right after the v1 freeze. The midterm review (9 November) asked for a stronger model, augmentation, live analysis, production infrastructure and a better interface. D48–D55 decide how, without breaking the rules that protect the results. v3.0 turns those decisions into owners and dates. The v2.2 draft was never adopted; its status content is folded in here.

## 2. What changed after the midterm

The professor approved the methodology (branches, commits, PRs, decisions) and raised the points below. Each one is now an approved decision (PR #111, 10 November 2026).

| Professor's point | Decision | What we do |
|---|---|---|
| The platform cannot be tested: it runs on mock data | D48 | Finish v1 to its freeze (F4) and replace the mock with the real model as soon as F4 is verified. |
| Move beyond the CNN: LSTM or voice-specific models | D49 | Three pre-declared v2 candidates: `cnn-aug` (control), `crnn` (CNN + BiLSTM), `aasist` (spoofing-specific). |
| FoR is too clean: augment for real-world audio | D50 | On-the-fly augmentation (noise, rooms, codecs, gain, RawBoost) for REAL and FAKE alike, disjoint from Stage 14. |
| Show and justify the confidence-score decisions | D55 | A decision board in the UI: calibrated p, threshold, uncertain band and how each was derived. |
| Live check during calls, with free tools | D51, D52 | Live microphone mode and `/api/live` (Must); the Chrome extension is not built (future work, D69 (1)). |
| Infrastructure: CPU/GPU, database, bottlenecks, justified limits | D53 | CPU process pool, `503` + `Retry-After` when busy, PostgreSQL, limits from a load test. |
| Staging before production | D54, D70 | dev → production (release tag + reviewer); staging removed by D70 after the professor withdrew the request (25 Nov). |
| The UI looks generic | D55 | Three design directions, one chosen, moderated usability test (SUS ≥ 68). |

The most important consequence: **Stage 13 now waits for v2** (D48). DEEP-VOICE can be used once, and D35 makes that run dataset-wide, so the only way to compare v1 and v2 on it is one joint run. If v2 is not frozen by **Friday 4 December, 23:59 Tunis**, Stage 13 runs with v1 and the baseline only, and v2 never gets an external result.

## 3. The project on one page

### What we build

A **decision-support** security control that analyses speech, from an uploaded recording or live during a call, and returns **authentic / synthetic / uncertain**, a confidence and a **LOW / MEDIUM / HIGH** risk level. It helps an analyst decide whether a high-risk request needs extra verification. It is not a biometric authenticator and not proof of fraud.

### Research questions and hypotheses

- **RQ1–RQ4 (Protocol §1):** FoR → DEEP-VOICE generalisation; CNN vs baseline; robustness under degradation; calibration under shift. H1–H7 unchanged.
- **H8 (D48):** model v2 generalises better than v1 to DEEP-VOICE (paired, speaker-clustered bootstrap; Stage 13).
- **H9 (D50):** augmentation transfers to a degradation family never used in training (white noise, 10 dB; Stage 14).

### Models

| Model | What | Status |
|---|---|---|
| `baseline-v1` | MFCC mean + std, RBF-SVM; evaluation only (D7, D37) | Fitted (Stage 7) |
| v1 CNN | 4 conv blocks on F3 log-Mel; seeds 0–2; risk policy on val-B (D42, D45) | Selected and fitted; frozen as F4 at Stage 12 |
| v2 (one of three) | `cnn-aug`, `crnn` or `aasist`, trained with D50 augmentation, selected on val-A (D49) | To build; frozen as F5 by 4 Dec |

**Deployment rule (D48 (5)):** v1 replaces the mock as soon as F4 is verified; v2 replaces v1 once F5 is verified, whatever its test results. Test results never choose the deployed model.

### Scope

**In scope**

- Baseline, v1 CNN, and v2 from the three named candidates
- Risk layer: Platt calibration, EER threshold, uncertain band (≤ 10% deferral on val-B)
- Live analysis: microphone mode, `/api/live` (Chrome extension: future work, D69 (1))
- FastAPI, React, PostgreSQL, Docker; production (no staging, D70)
- Robustness (Stage 14) and calibration analysis (Stage 15)

**Out of scope (future work)**

- XLS-R / wav2vec2 / WavLM / HuBERT / RawNet, ensembles
- Other features (LFCC, CQCC), explainability
- Mobile app; capture of native phone or in-app mobile calls (OS restriction, D51)
- Extra speech datasets; a second external dataset
- Adversarial defence, replay attacks

### Metrics (D4)

Positive class = spoof. Score level: EER, ROC-AUC. Decision level at the frozen threshold: accuracy, precision, recall, F1, forced-binary and on non-deferred cases. Always with the 3-way confusion matrix, coverage / deferral, class counts and 95% confidence intervals. Never accuracy alone. No t-DCF.

## 4. Hard rules

These rules protect the validity of the results. They are in `AGENTS.md`, so every AI tool follows them too, and the most important ones are enforced by tests in CI.

1. **DEEP-VOICE is touched exactly once:** one joint run of v1, v2 and the baseline after F5, or after the 4 December cut-off with v1 and the baseline only (D48, D35). Before it, only the D20 metadata accesses.
2. **Each ModelBundle is frozen as one unit** (checkpoint + preprocessing profile + risk policy), fitted on FoR only, before the external run: F4 for v1, F5 for v2.
3. **Scope lock:** baseline, v1 CNN, and v2 only from `cnn-aug`, `crnn`, `aasist` (D49). Live analysis only as D51–D52.
4. **The FoR test split is read once per model set** (D31, D41), through the reviewed runner, after the D46 gates. The v2 design was fixed before any test result (D48 (6)); a later change is post-hoc and labelled exploratory.
5. **Metrics as in D4.** Never accuracy alone.
6. **No silent changes** to data, splits, preprocessing, thresholds, metrics or scope. Propose, explain the impact, decide.
7. **Never commit audio, datasets, checkpoints or secrets.** Live audio is processed in memory and never stored or logged (D51).

## 5. How we work

### 5.1 Git workflow

- `main` is protected: pull request, 1 approval, required checks `checks`, `web` and `e2e`, branch up to date.
- One meaningful unit = one branch = one PR. Branch names `<type>/stage<N>-<name>` (D23); Conventional Commits.
- **Decision PRs:** all four approve in the PR, then the **approval commit** (statuses, AGENTS.md, protocol) goes on the branch **before** merge. Lesson from PR #67, which needed the record PR #112.
- **Numbers** (D-IDs, ADRs, log entries) are taken at merge time; the PR that merges second renumbers.
- After every merge: sync `main`, delete the branch locally and on GitHub, `git fetch --prune`, remove scratch files.

### 5.2 AI tools and human in the loop

- Claude (claude.ai) writes Claude Code prompts and reviews diffs line by line; each member drives their own tool (Claude Code, Codex, Antigravity). `AGENTS.md` is the single rulebook (D22).
- Ground truth over self-report: a claim counts only with real output (tests, data, hashes). Labels [verified], [doc], [hypothesis].
- AI never merges. AI assistance is recorded in each PR and in the project log.

### 5.3 Weekly rhythm

| When | What |
|---|---|
| Monday | Weekly sync (30 min): last week's gates, this week's cards, blockers, critical-path check. |
| Every day | Small PRs; reviews within 24 hours; a blocked card is said the same day. |
| Friday | Each owner updates the project log for their track; the plan's status column is updated by PR when a gate passes. |

### 5.4 Reproducibility and data

- Every result = pinned data + config file + code commit. Python 3.13 official; exact pins; CI on 3.13 with torch.
- Audio lives on the shared Drive only; datasets and the new noise/RIR corpora are pinned by hash (D21, D50 (5)).
- Every training run is logged to W&B with its config (D40). No audio, test or external data goes to W&B.

### 5.5 What a gate means

A step is closed when (1) its outputs are merged through a reviewed PR with green CI, (2) its review or sign-off is recorded, (3) the project log has an entry and an updated Current status, and (4) for a stage, its brief is merged and the teach-back is done (D32).

## 6. Team, ownership and reviews

| Member | Owns | This plan's main work | Reviews |
|---|---|---|---|
| Donia Mabrouk | Part C (Stages 7–12) · Track V lead | Stage 11 PR B, PR C, dry run, official run; F4; v2 candidates, training, selection, policy, v2 Stage 11; F5; P4 load test; Overview | Malak's V1 pipeline; Stage 14 |
| Malak Ben Salem | Part B (4–6) · V1 · Stage 14 | D50 corpora, splits and augmentation module; selection set S; Stage 14 for three model sets; D67 prototype corrections, HANDOFF-v3, B1 + shell, Flagged reviews | Part E (Iheb) |
| Khalil Dridi | Stages 13, 15, 16 · SRS v4.0 | Stage 13 runner for three model sets (D35, D36, D48); D52 warm-up analysis; Stages 15–16; SRS v4.0 draft; Analyse, Result, History, note filter, Live check screen, L1 contract review | Part C (Donia), required reviewer |
| Iheb Zemzemi | Part E (17–19) · Tracks L and P · SDD v2.0 | Real model in the API; PostgreSQL, process pool; production host (no staging, D70); live endpoint (L1); D68 designed, not shipped: threat-model and design notes by Fri 18 Dec (D69 (2)); Sign-in (D47) + Administration (System tab + Members note, D69 (2)) (UI redesign is shared, D67) | Track V code (with Khalil) |
| Everyone | Decision PRs · Stage 13 review · Stage 19 | Usability test sessions; report sections for own stages; final presentation | — |

> **Separation that protects the result.** The person who builds the models (Donia) never runs the external test. Khalil runs Stage 13 once, for all frozen models together; the result cannot be used to change any model, threshold or deployment choice.

### Hand-overs

| Hand-over | What is fixed in advance |
|---|---|
| Malak → Donia (V1 → V2) | Augmentation function signature, families, probabilities and seeds (D50); selection set S with hashes. |
| Donia → Iheb (F4, F5) | ModelBundle format and loader (verified hashes); analyzer interface unchanged; bundle version in every record. |
| Donia → Khalil (F4, F5) | Frozen bundles, `baseline-v1` hash, per-clip Stage 11 files; stored-score format per model (D36). |
| Iheb → all (P3) | Production host URL (team and usability test only until acceptance, D70 (3)), test accounts, demo audio set; never real personal recordings (D54 (8), D70 (4)). |

<!-- pagebreak -->

## 7. Timeline and critical path

Six weeks plus a short buffer. Weeks run Monday to Sunday. Dates are targets the owner commits to in the Monday sync; the **4 December cut-off is a rule**, not a target (D48 (4)).

| Week | Dates | Track R (v1) | Track V (v2) | Part D | Tracks L and P |
|---|---|---|---|---|---|
| W1 | 9–15 Nov | PR B, PR C | V1 corpora pinned, augmentation module | Stage 13 runner: three model sets | UI directions and vote; PostgreSQL; process pool; hosting chosen |
| W2 | 16–22 Nov | Dry run → official Stage 11 → F4 | V2 candidate code; V3 training starts | Runner review | Real model in the API (after F4); hosting (late, P3); UI build starts |
| W3 | 23–29 Nov | Stage 11 results, brief, teach-back | V3 search by relay; selection (target Fri 27, estimate Sat 28) | D52 warm-up analysis (val-B) | L1 contract in SDD v2.0 by Sun 29; D67 prototype corrections, HANDOFF-v3, B1 + shell |
| W4 | 30 Nov–6 Dec | — | Seeds 1–2; risk-v2; V4 v2 Stage 11; **V5 F5 by Fri 4 Dec** | **Stage 13 joint run (Sat 5–Sun 6 Dec)** | Production host Mon 30–Tue 1 (after the D54 amendment; no staging, D70); note filter and L1 contract review by Tue 1; D63 Wed 2; L1 from Thu 3; F5 swap by Sun 6 if the winner takes log-Mel input and the bundle reaches Iheb Sat 5 morning (D69 (6)); `tools/loadtest/` PR Sat 5–Sun 6 (Donia, D69 (5)) |
| W5 | 7–13 Dec | — | — | Stages 14, 15, 16 | P4 part 1 (uploads) Mon 7–Tue 8; L1 on the production host Wed 9–Mon 14, then P4 part 2 (date set then); Phase 1 screens by Thu 10; usability test Fri 11–Sun 13 if L1 is on the production host or the local `vsd-real` stack by Thu 10 (D69 (3), D70); production Sun 13 if L1, the F5 swap and the P4 decision are done, otherwise right after them |
| W6 | 14–20 Dec | Report | Report | Report | Usability test Mon 14–Tue 15 if not in W5; Phase 2; D68 notes (designed, not shipped, D69 (2)); feature freeze Fri 18 (D69 (4)); report, demo set, two rehearsals |
| Buffer | 21–23 Dec |  |  |  | Final checks; submission by Wed 23 Dec (target) |

### Critical path to the cut-off

| Step | Owner | Target date | Depends on |
|---|---|---|---|
| Noise and RIR corpora: licence, hash, 50/50 split pinned | Malak | Fri 13 Nov | — |
| Augmentation module + selection set S merged | Malak | Wed 18 Nov | corpora |
| v2 candidate code (`cnn-aug`, `crnn`, `aasist`) merged; batch-size benchmark in `configs/` | Donia | Fri 20 Nov | augmentation module |
| Training (seed 0 per candidate) and selection on val-A | Donia | Fri 27 Nov; estimate Sat 28 (25 Nov) | candidate code |
| Seeds 1–2 of the winner; v2 risk policy on val-B | Donia | Mon 30 Nov | selection |
| Runner accepts the v2 model set; v2 Stage 11 run (once) | Donia | Wed 2 Dec | policy |
| **F5 verified by the bundle loader** | Donia | **Fri 4 Dec, 23:59** | v2 Stage 11 |
| Stage 13 joint run (v1, v2, baseline) | Khalil | Sat 5–Sun 6 Dec | F5 or cut-off; Stage 13 runner |

> **Cut line, if a week slips.** Cut in this order: (1) the Chrome extension: **cut 26 Nov (D69 (1))**, live mic mode covers live analysis; (2) the `aasist` candidate: `crnn` still answers the temporal-model request (decided 24 Nov: kept, rehearsal 29.5 h ≤ 95 h, D48 (6)); (3) D68 (email sign-in, roles, Members): **cut 26 Nov, designed, not shipped (D69 (2))**, D47 stays; (4) the usability test shrinks to 3 participants. Never cut: the real model in the platform, the single-run rules, augmentation, Stage 11 and Stage 13. After the feature freeze (Fri 18 Dec, D69 (4)): only fixes, the report and the two rehearsals.

## 8. Track R — v1 to its freeze (Stages 11–12) · Donia

#### 11 · FoR internal test, v1 (once) — DONE (Entries 161, 163)

- **Done:** D41 registry and per-clip storage (PR #62); D46 procedure (PRs #67, #112); runner, PR A (PR #68).
- **Tasks:**
  - **PR B:** report computed only from the stored per-clip files: D4 metrics with class counts per subgroup, H1 with bootstrap CI, H5, seed mean ± SD.
  - **PR C:** notebook 06: environment gate and record, reproduction gate, rehearsal, claim, Drive flush and remount, registry SHA-256 against `claim.json`, `run_id` from `claim.json` in code, score, export.
  - **Dry run on T4**, validation splits only (MFCC-on-T4 still unverified), then the freeze checklist.
  - **Official run once**; results PR; registry committed byte for byte (D46 (6)); brief and teach-back.
- **Dates:** PR B Fri 13 Nov · PR C Mon 16 Nov · dry run Tue 17 Nov · official run Wed 18–Thu 19 Nov.
- **Gate:** One completed `stage11_test` claim for the v1 model set; results and brief merged; Khalil's teach-back.

#### 12 · ModelBundle v1 freeze (F4) and hand-over — DONE (F4 12 Nov, Entries 168, 174)

- **Tasks:**
  - Assemble F4: bundle-seed checkpoint + F3 profile + `risk-v1` + manifest; record `baseline-v1` hash (D36).
  - Verify with the bundle loader; hand over to Iheb for the API (P1) and to Khalil for Stage 13.
- **Dates:** F4 verified Sun 22 Nov.
- **Gate:** Loader verifies every part; F4 manifest SHA-256 recorded in the log; the mock can be removed.

## 9. Track V — model v2 (V1–V5) · Donia, with Malak

#### V1 · Corpora and augmentation pipeline (D50) — DONE (S and resource cache built 22 Nov; evidence Entry 215)

- **Owner:** Malak; review Donia.
- **Tasks:**
  - Noise (MUSAN, OpenSLR 17) and RIR (RIRS_NOISES, OpenSLR 28): licence, download hash, seeded 50/50 split by file into training and selection halves; pins in `configs/`.
  - Augmentation module on the waveform before F3: five families with the D50 values; REAL and FAKE alike; probability 0.8; no white noise; no §9 MP3 bitrates or low-pass family; ffmpeg and codecs pinned.
  - Selection set S built once from val-A (seed 2026): clean, noise 10 dB, reverberation, Opus 16 kbps, narrowband; hashes stored.
  - Tests on synthetic audio: determinism with a seed, family probabilities, peak limit, disjointness checks.
- **Dates:** Corpora Fri 13 Nov · module and S merged Wed 18 Nov.
- **Gate:** Merged with tests; pins and hashes recorded; Donia's review.

#### V2 · Candidate models — DONE (Entry 170; training code, benchmark and resume PRs #129, #130, #137)

- **Tasks:**
  - `cnn-aug` reuses the v1 network (D42); `crnn` and `aasist` as specified in D49 (`aasist` from a reference implementation pinned by commit, F3 waveform input).
  - Unit tests (shapes, one logit, determinism); short compute benchmark → batch sizes in `configs/` before training.
- **Dates:** Merged Fri 20 Nov. Review Khalil and Iheb.

#### V3 · Training and selection on val-A — NOW (rehearsal done 23 Nov; search running since 24 Nov, `cnn-aug` epoch 25 saved 25 Nov)

- **Tasks:**
  - Seed 0 per candidate, max 30 epochs, early stopping on the mean val-A EER over S (patience 5); W&B for every run.
  - Winner = lowest mean; ties: fewer parameters, then `cnn-aug`, `crnn`, `aasist`. Seeds 1 and 2 for the winner; bundle seed = median.
  - Relay from `d5ffe18`: sessions rotate between team members' own free accounts, one at a time.
- **Dates:** Selection Fri 27 Nov (may slip to Sat 28) · winner seeds 1–2 (≈ 12 h) Sat 28 – Mon 30 Nov · then the search evidence PR. Estimate of 25 Nov [stated by Donia]: about 9 saved epochs per session; seed-0 remainder ≈ 12.75 h (~7 sessions).
- **Gate:** Selection results committed (validation only); every candidate's val-A table reported.

#### V4 · v2 risk policy and v2 Stage 11 — CODE DONE (PRs #141, #142, #150; notebooks 09–10); runs after V3

- **Tasks:**
  - Protocol §6 method unchanged on clean val-B, bundle seed only (D45 rule).
  - Runner extended to the v2 model set (small PR); one `stage11_test` claim for v2, all three seeds stored.
  - Closure (D32, no amendment): brief after the results → v2 section in `docs/stages/stage11-for-test.md` → teach-back (Khalil required) → closeout, as v1 (Entry 163); agreed by Iheb, Khalil and Malak.
- **Dates:** Notebook 10 (risk-v2) ~Mon 30 Nov – Tue 1 Dec · notebook 09 dry run ~Tue 1 Dec · official run Wed 2 Dec. Both notebooks operated by Donia only.

#### V5 · ModelBundle v2 freeze (F5) — CODE DONE (PRs #138, #144); build after V4

- **Tasks:**
  - Assemble and verify F5 with the existing loader; hand over to Iheb (deployment) and Khalil (Stage 13).
- **Dates:** build Thu 3 Dec · **Fri 4 Dec, 23:59 Tunis: cut-off (D48 (4)).**
- **Gate:** F5 verified before the cut-off, or v2 is excluded from Stage 13 and that is logged.

## 10. Part D — joint external run and analyses (Stages 13–16) · Khalil, Malak

#### 13 · DEEP-VOICE external run (once, guarded) — TO DO

- **Prepare:**
  - Stage 13 runner (D35) for three model sets in one pass: v1 bundle CNN, v2 bundle model, `baseline-v1`; one stored-score file per model with identical columns (D36); D17 filter, D19 per-speaker table, D33, D34 file-level view.
  - Governance allowlist decision before the run (Protocol §8 open item); durable claim beyond Colab (D35).
  - D52 warm-up analysis on simulated calls from val-B (descriptive).
- **Run:** Sat 5–Sun 6 Dec, after F5 or the cut-off. Review by all four.
- **Report:** H2, H8 (paired v1–v2), RQ2 (CNN vs baseline), the live rule's end-of-call accuracy via D34, 87.5% FAKE share and RVC-only scope stated.

#### 14 · Robustness (FoR test only) — v1 FOLLOW-UP MERGED (#164, Entry 216); official run waits for the all-four H6 approval

- **Owner:** Malak; review Donia.
- **Tasks:**
  - Complete §9 grid for `baseline-v1`, v1 bundle CNN and v2 bundle model (one `stage14_robustness` claim each), reusing each model's Stage 11 clean scores (D41).
  - For v2, label conditions: white noise = unseen family, MP3 = seen family / unseen bitrates, bandwidth = seen family (D50 (3)). H4, H6 (v1), H9.
- **Dates:** v1 official run after notebook 09 on Wed 2 Dec, unless its registry evidence is merged by Tue 1 Dec; registry writers never overlap (Entry 216). v2 from Mon 7 Dec.

#### 15–16 · Calibration and error analysis — TO DO

- **Owner:** Khalil; review Donia.
- **Tasks:**
  - Reliability diagrams and ECE (10 bins) on FoR test and stored DEEP-VOICE scores; H3; deferral under shift (RQ4).
  - Errors by subgroup, speaker, confidence and deferral; H5, H7, D43 (v1; descriptive for v2).
- **Dates:** Mon 7–Sun 13 Dec.

## 11. Track L — live analysis (L1–L3) · Iheb

#### L1 · Streaming endpoint `/api/live` — TO DO

- **Tasks:**
  - WebSocket through nginx, session required; mono PCM at native rate; server-side F3 including `soxr_hq` (D52 (1)).
  - Non-overlapping 2 s windows; D17 speech filter; per-window calibrated p; running verdict = D34 rule on the stream; warm-up 3 kept windows; trend of the last 5 windows without a label.
  - Same process pool and limits as uploads (D53); per-message size cap; no audio stored or logged; one metadata record per session.
- **Dates:** The `/api/live` contract in SDD v2.0 by Sun 29 Nov, reviewed by Khalil by Tue 1 Dec · built from Thu 3 Dec (5–8 d; the F5 swap comes first if the winner takes log-Mel input, D69 (6)) · on the production host Wed 9 – Mon 14 Dec (D70).

#### L2 · Live microphone mode (web app) — TO DO

- **Tasks:**
  - Start/Stop, visible listening indicator, consent and decision-support notice (D51); live screen = the D67 Live check screen (Khalil), built against the L1 contract and a mock.
- **Dates:** From Mon 30 Nov; by Thu 10 Dec.

#### L3 · Chrome extension (Should) — NOT BUILT (D69 (1): future work)

- **Tasks:**
  - Design first: authentication against D47 (SameSite=Strict cookie, Origin check) decided and security-reviewed in SDD v2.0.
  - Manifest V3, `tabCapture` after a user click, offscreen document, tab audio routed back to the speakers; same endpoint; tested on WhatsApp Web and Google Meet.
- **Status:** not built for delivery; recorded as future work (D69 (1)). The tasks above are the design notes for that future work.

## 12. Track P — platform and production (P1–P5, Stage 19) · Iheb

#### P1 · Real model in the API — DONE (#165, Entry 217)

- **Tasks:**
  - Replace `MockAnalyzer` with the verified F4 bundle behind the existing analyzer interface; versions in every record (NFR-07, NFR-09).
  - End-to-end tests with the real bundle; later swap to F5 by the deployment rule (D48 (5)).
- **Dates:** F5 swap on the production host by Sun 6 Dec if the winner takes log-Mel input and the verified bundle reaches Iheb by Sat 5 Dec morning, reviewed by Donia (D69 (6)); otherwise with L1, from Thu 3 Dec.

#### P2 · PostgreSQL and process pool (D53) — DONE (P2a #155, P2b #162; provisional K = 2)

- **Tasks:**
  - PostgreSQL behind the repository interface for analyses, notes and D47 sessions; migration; CI service container.
  - Bounded process pool (kills timed-out work); `503` + `Retry-After` when full; no broker.
- **Dates:** W1–W2.

#### P3 · Production (D54; no staging, D70) — LATE (D54 (9) hosting amendment in progress, not merged)

- **Tasks:**
  - Hosting provider chosen and recorded (one free VM, one Compose project, D70 (5)); images to GHCR once per `main` commit; before each release tag the images run on the dev `vsd-real` stack from GHCR by that SHA (D70 (2)); production from a release tag with a required reviewer; HTTPS; smoke test and rollback; until production acceptance the host serves only the team and the usability test (D70 (3)).
- **Dates:** Iheb's order (25 Nov; D69 (6), D70), after the D54 amendment: production host Mon 30 Nov – Tue 1 Dec · D63 Wed 2 Dec · F5 swap by Sun 6 Dec if the winner takes log-Mel input · L1 from Thu 3 Dec · production acceptance Sun 13 Dec if L1, the F5 swap and the P4 decision are done, otherwise right after them.

#### P4 · Load test and limits (D53 (3)) — TO DO

- **Owner:** Donia (from Iheb, agreed 25 Nov).
- **Tasks:**
  - Iheb's brief; `tools/loadtest/` PR; reference workload on the production host with the deployed bundle, before production acceptance (D70 (3)); every result file records the loaded model version and bundle hash from `/status` (D69 (5)).
  - Part 1 (uploads): measurements only. Part 2 (live sessions: L, `analysis_timeout_s`, `L_MAX`) after L1 is on the production host, date set then.
  - One decision records K, L, `analysis_timeout_s`, `L_MAX`, tmpfs, Retry-After, the worker start timeout and the Q1 decoded-size bound after part 2, before production acceptance (SDD v2.0 §7.4, D53 (3), NFR-15).
  - Values are decided on the deployed bundle (F5 by D48 (5)). If F5 is not on the production host by Mon 7 Dec, part 1 runs on F4 and its upload measurements are repeated on F5 in part 2. If F5 misses Fri 4 Dec, F4 stays deployed (D48 (4)) and both parts run on F4.
- **Dates:** Brief by Mon 30 Nov · `tools/loadtest/` PR Sat 5 – Sun 6 Dec (Donia, reviewed by Iheb; no dry run, D69 (5)) · part 1 Mon 7 – Tue 8 Dec · part 2 after L1 is on the production host.

#### P5 · Interface redesign and usability (D55) — DIRECTION CHANGED (D67, approved 25 Nov); identity and roles D68 designed, not shipped (D69 (2))

- **Owner:** shared (D67 split, Entry 219).
- **Tasks, Phase 1 (usability-test screens, by Thu 10 Dec):**
  - Malak: prototype corrections D67 (1)–(7) (1 d); HANDOFF-v3 (2 d, Phase 1 parts first); B1 + shell (2–3 d).
  - Khalil: note filter `note=missing|present` (0.5 d) and L1 contract review (1.5 d) by Tue 1 Dec; Analyse, Result, History (3 d) and Live check screen (1 d) by Thu 10 Dec.
- **Tasks, Phase 2 (by Fri 18 Dec):**
  - Malak: Flagged reviews (2 d). Donia: Overview (1.5–2 d, from Wed 9 Dec; fallback Khalil from Mon 14).
  - Iheb: the "D68: designed, not shipped" subsection of `docs/threat_model.md` (0.5–1 d) by Fri 18 Dec; Administration = System tab plus a Members tab stating that accounts are created by an administrator on the server (D69 (2)).
- **Kept (D55, D67):** §11 fields, NFR-05 wording, axe scans, decision board, per-window timeline, live screen.
- **Usability test (D55 (4)):** moderated, 3–5 people outside the team, fixed task script, SUS ≥ 68; findings and fixes logged; on D47 username sign-in.
- **Dates:** Phase 1 Thu 10 Dec · usability test Fri 11 – Sun 13 Dec if L1 is on the production host or the local `vsd-real` stack by Thu 10 Dec (D69 (3), D70), otherwise Mon 14 – Tue 15 Dec · Phase 2 Fri 18 Dec (feature freeze, D69 (4)).
- **Estimate (owners, 25 Nov):** 16.5–21 working days, excluding the L1 contract review, plus 3 days of prototype corrections and HANDOFF-v3.

#### 19 · Integration, report and delivery — TO DO

- **Tasks:**
  - SRS v4.0 (Khalil, W2) and SDD v2.0 (Iheb, W3; the `/api/live` contract in SDD v2.0 by Sun 29 Nov); README and setup current; fresh-machine test (NFR-10).
  - Final report: each owner writes their stages; demo audio set; two full rehearsals.
- **Dates:** W5–W6; submission target Wed 23 Dec.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| v2 training does not fit the free T4 time | One seed per candidate during selection; 30-epoch cap; batch benchmark in W1; cut line drops `aasist` first; cut-off protects Stage 13. |
| Stage 11 reproduction gate fails (feature bytes differ) | Dry run first on validation splits; a mismatch stops Stage 11 for a logged decision before any test access (D46 (4)). |
| Test results tempt a change to v2 | v2 design approved before any test result (D48 (6)); deployment by rule (D48 (5)); registry refuses reruns. |
| Iheb's tracks are the heaviest | Extension and D68 build cut, staging removed (D69, D70); Khalil writes SRS v4.0; UI directions prepared by Claude; reviews spread. |
| Enterprise scope larger than Direction D | Shared build (D67 split); read side reuses `/stats` and notes; D68 designed, not shipped (D69 (2)); Iheb's load 20–27 days against 20; production may move after Mon 14 Dec; delivery Wed 23 Dec is the limit. |
| Extension cannot authenticate safely | Not built for delivery (D69 (1)); the design question stays in the L3 future-work notes. |
| Free Colab GPU quota stops long T4 sessions (seen 22–23 Nov) | Rehearsal and search resume per epoch on Drive and are bound to the commit and input hashes, not to an account: sessions rotate between team members' own accounts, one at a time; a second account of the same person is not used (Colab FAQ); the rehearsal's measured cut line decides whether `aasist` is dropped (§7). |
| PR merged by its author without review (#159, #160) | Reviewer merges after the diff review (§5.1); a post-merge review blocks any irreversible step (registry claim, freeze) until a reviewed follow-up is merged. |
| Merge before approval commit | Approval commit on the branch before merge (Section 5.1); record PR otherwise, as #112. |
| AI-generated errors accepted unchecked | Line-by-line review, real test output, CI with torch; ground truth over self-report. |

## 14. Open items (W3–W4, from 26 Nov)

1. Donia: search sessions and selection (target Fri 27 Nov, estimate Sat 28); search evidence PR; notebooks 10 and 09; F5 by Fri 4 Dec 23:59; P4 (Iheb's brief Mon 30, `tools/loadtest/` PR Sat 5 – Sun 6, D69 (5)); Overview from Wed 9 Dec.
2. Malak: prototype corrections D67 (1)–(7); HANDOFF-v3 (Phase 1 parts first); B1 + shell; Stage 14 v1 official run after the all-four approval (§10 dates); Stage 14 compute estimate (D69, Entry 222).
3. Khalil: note filter and L1 contract review by Tue 1 Dec; Analyse, Result, History and Live check by Thu 10 Dec; search sessions; Stage 13 runner review for the Sat 5 – Sun 6 Dec run (status to confirm).
4. Iheb: D54 (9) hosting amendment (one free VM, D70 (5); date to set); P4 brief by Mon 30 Nov; production host Mon 30 Nov – Tue 1 Dec; D63 Wed 2 Dec; F5 swap by Sun 6 Dec if the winner takes log-Mel input (D69 (6)); L1 from Thu 3 Dec; the `/api/live` contract in SDD v2.0 by Sun 29 Nov; D68 notes by Fri 18 Dec.
5. All: all-four H6 approval for Stage 14 before any Stage 14 test access; usability test W5 on the production host or `vsd-real` (D69 (3), D70) (or Mon 14 – Tue 15 Dec, §12 P5).
6. Dates still missing: hosting provider and the production host date (after the D54 amendment).

<!-- pagebreak -->

## Appendix A — Decisions since v2.1

Authoritative text: `docs/decisions/README.md`. Decisions D1–D25 are summarised in v2.1, Appendix A.

| ID | Decision |
|---|---|
| D26 | Work distribution (Execution Plan v2.1): parts A–E, owners, consumer reviews producer. |
| D27 | Shared AI tooling: one skill (`pr-prep`) and one MCP server (Context7). |
| D28 | Dataset freeze F2: FoR manifest and hash ledger. |
| D29–D31 | Baseline threshold at its own val-B EER point; CNN search space; Stage 11 runs once (registry). |
| D32 | Stage briefs and teach-back are part of stage closure. |
| D33–D36 | Stage 13 reporting conventions, file-level view, guarded access, baseline in the single external run. |
| D37–D40 | Baseline tuning; React replaces Streamlit (ADR 0008); F3 preprocessing profile; W&B from Stage 8. |
| D41 | Registry purpose key; paired clean scores for Stage 14 (ADR 0010). |
| D42–D45 | CNN design; loudness diagnostic; Platt stop rule; calibrated output for the bundle seed only. |
| D46 | Stage 11 run procedure: T4, CPU scoring, environment gate and record, reproduction gate, durable claim, rerun rule. |
| D47 | Analyst sign-in: username + password, server-side cookie sessions (ADR 0011). |
| D48 | v2 track; one joint external run; cut-off 4 Dec; deployment by rule (ADR 0012). |
| D49–D50 | v2 candidates and selection; training augmentation disjoint from Stage 14 (ADR 0013). |
| D51–D52 | Live analysis scope; live decision rule = D34 on the stream (ADR 0014). |
| D53–D54 | CPU process pool, PostgreSQL, measured limits; dev/staging/production (ADR 0015); staging removed by D70. |
| D55 | Interface redesign and usability evidence. |
| D56 | RawBoost algorithm 1 and uniform sampling over seven codec options (amends D50). |
| D57 | Stage 11 reporting conventions. |
| D58 | Processor-bound reproduction gate for the baseline MFCC; runtime 2026.10 pinned (amends D46). |
| D59 | ModelBundle layout for the two-file checkpoint. |
| D60 | ModelBundle v1 freeze F4. |
| D61 | Model v2 candidate implementation details (clarifies D49, D59 (7)). |
| D62 | Not used. |
| D63 | Calibrated probability, policy and window fields in the API (amends D26, D55; ADR 0016). |
| D64 | Selection-set recipe re-frozen under the declared environment. |
| D65 | UI Direction D (ADR 0017); direction changed by D67. |
| D66 | Reserved for F5 (written at the freeze). |
| D67 | UI direction VoiceCheck Enterprise, with seven corrections (amends D65; ADR 0018). |
| D68 | Email sign-in, server-side roles and member management; designed, not shipped (D69 (2)) (amends D47; ADR 0019). |
| D69 | Delivery scope: extension not built, D68 designed not shipped, feature freeze Fri 18 Dec, no P4 dry run, F5 swap may come before L1 (amends D67, D68). |
| D70 | Staging removed: dev and production; load test on the production host before acceptance (amends D53, D54; the professor's request withdrawn, 25 Nov). |
| H7–H9 | Pauses (Stage 16); v2 vs v1 on DEEP-VOICE (Stage 13); unseen white-noise family (Stage 14). |

## Appendix B — Freeze points and registry claims

| Freeze / claim | Content | When |
|---|---|---|
| F1 | Experimental Protocol v1.0 (+ addendum, PR #111) | Stage 3 (done) |
| F2 | Split, manifests and hash ledger (D28) | Stage 5 (done) |
| F3 | Preprocessing profile (D39) | Stage 6 (done) |
| F4 | ModelBundle v1 + `baseline-v1` hash (D36) | Stage 12, target Sun 22 Nov |
| F5 | ModelBundle v2 (D49) | V5, by Fri 4 Dec 23:59 |
| `stage11_test` | One claim per model set: v1 set, then v2 set (D31, D41, D48 (7)) | W2 and W4 |
| `stage14_robustness` | One claim per model set: baseline, v1, v2 (D41, D48 (8)) | W3–W5 |
| Stage 13 | One dataset-wide claim: all frozen models in one pass (D35, D36, D48) | Sat 5–Sun 6 Dec |

## Appendix C — Document versions

| Document | Version now | Next | Owner, target |
|---|---|---|---|
| Decision register | D1–D55 | New rows as needed | Decision owner |
| Experimental Protocol | v1.0 + addendum (PR #111) | Further addenda only by decision | Donia |
| SRS | v3.0 | v4.0: live analysis (WH-01 changed), infrastructure, priorities | Khalil, W2 |
| SDD | v1.0 | v2.0: streaming, process pool, PostgreSQL, environments, extension security | Iheb, W3 |
| Execution Plan | v3.0 (this) | Status updates by PR | Donia |

## Appendix D — What changed since v2.1

| v2.1 said | v3.0 does |
|---|---|
| One deployed CNN; augmentation out of scope (D10) | v1 CNN kept; v2 from three candidates with augmentation (D49, D50). |
| Stage 13 right after the v1 freeze | Stage 13 waits for F5 or the 4 Dec cut-off; one joint run (D48). |
| Streamlit dashboard | React + TypeScript web app (D38); redesign and usability test (D55). |
| Upload-only service; live detection out of scope | Live mic mode and `/api/live`; Chrome extension (D51, D52; not built, D69). |
| SQLite; one environment | PostgreSQL, process pool, measured limits; production (D53, D54; staging removed, D70). |
| 14-week targets | Dated six-week plan with a critical path and cut line. |
| Khalil (surname not recorded) | Khalil Dridi; owns SRS v4.0 and the joint Stage 13 run. |

---

*Prepared from the decision register (D1–D55), Protocol v1.0 and its addendum, the project log (Entries 1–154), Execution Plan v2.1, the v2.2 draft, SRS v3.0 and SDD v1.0, with AI assistance (Claude); to be reviewed by the team.*
