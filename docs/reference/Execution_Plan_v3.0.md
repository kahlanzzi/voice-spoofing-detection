# Voice-Spoofing Detection for Fraud Prevention

## Execution Plan v3.0

*Execution Plan v3.0: from the midterm to delivery, with model v2, live analysis and production, stage by stage, and who owns what*

- **Team:** Donia Mabrouk · Iheb Zemzemi · Malak Ben Salem · Khalil Dridi
- **Version / date:** v3.0 · 10 November 2026 (replaces v2.1 and the unadopted v2.2 draft)
- **Repository:** github.com/kahlanzzi/voice-spoofing-detection
- **Controlling documents:** Decision register (D1–D55) › SRS v3.0 (v4.0 in preparation) › SDD v1.0 (v2.0 in preparation) › Protocol v1.0 + addendum › this plan › Proposal

### Where we are today

| | Item | Detail |
|---|---|---|
| **DONE** | Stages 0–10 | Repository, audit, threat model, Protocol v1.0 (F1), split and freeze (F2), preprocessing (F3), baseline, CNN (3 seeds), risk policy on val-B. Briefs 00–10 accepted. |
| **DONE** | Mock MVP | Stage 17 API and Stage 18 React app merged (PRs #77–#109): sign-in, upload, result, History, status, Docker/nginx. Runs on a labelled MockAnalyzer. |
| **DONE** | Stage 11 ready | D46 run procedure approved (PRs #67, #112); runner merged (PR #68). |
| **NEW** | After the midterm | Professor's feedback (9 Nov) turned into D48–D55, approved in PR #111: model v2, one joint external run, live analysis, production infrastructure, UI redesign. |
| **NEXT** | This week | Stage 11 PR B and PR C · D50 corpora and augmentation · UI directions · PostgreSQL and process pool · Stage 13 runner for three model sets. |
| **DATE** | Hard dates | v2 freeze (F5) cut-off: Fri 4 Dec 2026, 23:59 Tunis (D48). Delivery target: Wed 23 Dec 2026, to be confirmed against the course calendar. |

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
- 14\. Open items this week
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
| Live check during calls, with free tools | D51, D52 | Live microphone mode and `/api/live` (Must); Chrome extension for web calls (Should). |
| Infrastructure: CPU/GPU, database, bottlenecks, justified limits | D53 | CPU process pool, `503` + `Retry-After` when busy, PostgreSQL, limits from a load test. |
| Staging before production | D54 | dev → staging (auto after green CI) → production (release tag + reviewer). |
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
- Live analysis: microphone mode, `/api/live`, Chrome tab-capture extension
- FastAPI, React, PostgreSQL, Docker; staging and production
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
| Donia Mabrouk | Part C (Stages 7–12) · Track V lead | Stage 11 PR B, PR C, dry run, official run; F4; v2 candidates, training, selection, policy, v2 Stage 11; F5 | Malak's V1 pipeline; Stage 14 |
| Malak Ben Salem | Part B (4–6) · V1 · Stage 14 | D50 corpora, splits and augmentation module; selection set S; Stage 14 for three model sets | Part E (Iheb) |
| Khalil Dridi | Stages 13, 15, 16 · SRS v4.0 | Stage 13 runner for three model sets (D35, D36, D48); D52 warm-up analysis; Stages 15–16; SRS v4.0 draft | Part C (Donia), required reviewer |
| Iheb Zemzemi | Part E (17–19) · Tracks L and P · SDD v2.0 | Real model in the API; PostgreSQL, process pool, load test; staging/production; live endpoint, live UI, extension; UI redesign | Track V code (with Khalil) |
| Everyone | Decision PRs · Stage 13 review · Stage 19 | Usability test sessions; report sections for own stages; final presentation | — |

> **Separation that protects the result.** The person who builds the models (Donia) never runs the external test. Khalil runs Stage 13 once, for all frozen models together; the result cannot be used to change any model, threshold or deployment choice.

### Hand-overs

| Hand-over | What is fixed in advance |
|---|---|
| Malak → Donia (V1 → V2) | Augmentation function signature, families, probabilities and seeds (D50); selection set S with hashes. |
| Donia → Iheb (F4, F5) | ModelBundle format and loader (verified hashes); analyzer interface unchanged; bundle version in every record. |
| Donia → Khalil (F4, F5) | Frozen bundles, `baseline-v1` hash, per-clip Stage 11 files; stored-score format per model (D36). |
| Iheb → all (P3) | Staging URL, test accounts, demo audio set; never real personal recordings (D54 (8)). |

<!-- pagebreak -->

## 7. Timeline and critical path

Six weeks plus a short buffer. Weeks run Monday to Sunday. Dates are targets the owner commits to in the Monday sync; the **4 December cut-off is a rule**, not a target (D48 (4)).

| Week | Dates | Track R (v1) | Track V (v2) | Part D | Tracks L and P |
|---|---|---|---|---|---|
| W1 | 9–15 Nov | PR B, PR C | V1 corpora pinned, augmentation module | Stage 13 runner: three model sets | UI directions and vote; PostgreSQL; process pool; hosting chosen |
| W2 | 16–22 Nov | Dry run → official Stage 11 → F4 | V2 candidate code; V3 training starts | Runner review | Real model in the API (after F4); staging live; UI build starts |
| W3 | 23–29 Nov | Stage 11 results, brief, teach-back | V3 selection; V4 policy fit | D52 warm-up analysis (val-B) | `/api/live` and live mic mode; extension prototype |
| W4 | 30 Nov–6 Dec | — | V4 v2 Stage 11; **V5 F5 by Fri 4 Dec** | **Stage 13 joint run (Sat 5–Sun 6 Dec)** | Extension on WhatsApp Web; load test → limits |
| W5 | 7–13 Dec | — | — | Stages 14, 15, 16 | Production deploy; usability test → fixes |
| W6 | 14–20 Dec | Report | Report | Report | Report, demo set, two rehearsals |
| Buffer | 21–23 Dec |  |  |  | Final checks; submission by Wed 23 Dec (target) |

### Critical path to the cut-off

| Step | Owner | Target date | Depends on |
|---|---|---|---|
| Noise and RIR corpora: licence, hash, 50/50 split pinned | Malak | Fri 13 Nov | — |
| Augmentation module + selection set S merged | Malak | Wed 18 Nov | corpora |
| v2 candidate code (`cnn-aug`, `crnn`, `aasist`) merged; batch-size benchmark in `configs/` | Donia | Fri 20 Nov | augmentation module |
| Training (seed 0 per candidate) and selection on val-A | Donia | Fri 27 Nov | candidate code |
| Seeds 1–2 of the winner; v2 risk policy on val-B | Donia | Mon 30 Nov | selection |
| Runner accepts the v2 model set; v2 Stage 11 run (once) | Donia | Wed 2 Dec | policy |
| **F5 verified by the bundle loader** | Donia | **Fri 4 Dec, 23:59** | v2 Stage 11 |
| Stage 13 joint run (v1, v2, baseline) | Khalil | Sat 5–Sun 6 Dec | F5 or cut-off; Stage 13 runner |

> **Cut line, if a week slips.** Cut in this order: (1) the Chrome extension: live mic mode still covers live analysis; (2) the `aasist` candidate: `crnn` still answers the temporal-model request; (3) the usability test shrinks to 3 participants. Never cut: the real model in the platform, the single-run rules, augmentation, Stage 11 and Stage 13.

## 8. Track R — v1 to its freeze (Stages 11–12) · Donia

#### 11 · FoR internal test, v1 (once) — NOW

- **Done:** D41 registry and per-clip storage (PR #62); D46 procedure (PRs #67, #112); runner, PR A (PR #68).
- **Tasks:**
  - **PR B:** report computed only from the stored per-clip files: D4 metrics with class counts per subgroup, H1 with bootstrap CI, H5, seed mean ± SD.
  - **PR C:** notebook 06: environment gate and record, reproduction gate, rehearsal, claim, Drive flush and remount, registry SHA-256 against `claim.json`, `run_id` from `claim.json` in code, score, export.
  - **Dry run on T4**, validation splits only (MFCC-on-T4 still unverified), then the freeze checklist.
  - **Official run once**; results PR; registry committed byte for byte (D46 (6)); brief and teach-back.
- **Dates:** PR B Fri 13 Nov · PR C Mon 16 Nov · dry run Tue 17 Nov · official run Wed 18–Thu 19 Nov.
- **Gate:** One completed `stage11_test` claim for the v1 model set; results and brief merged; Khalil's teach-back.

#### 12 · ModelBundle v1 freeze (F4) and hand-over — NEXT

- **Tasks:**
  - Assemble F4: bundle-seed checkpoint + F3 profile + `risk-v1` + manifest; record `baseline-v1` hash (D36).
  - Verify with the bundle loader; hand over to Iheb for the API (P1) and to Khalil for Stage 13.
- **Dates:** F4 verified Sun 22 Nov.
- **Gate:** Loader verifies every part; F4 manifest SHA-256 recorded in the log; the mock can be removed.

## 9. Track V — model v2 (V1–V5) · Donia, with Malak

#### V1 · Corpora and augmentation pipeline (D50) — NEXT

- **Owner:** Malak; review Donia.
- **Tasks:**
  - Noise (MUSAN, OpenSLR 17) and RIR (RIRS_NOISES, OpenSLR 28): licence, download hash, seeded 50/50 split by file into training and selection halves; pins in `configs/`.
  - Augmentation module on the waveform before F3: five families with the D50 values; REAL and FAKE alike; probability 0.8; no white noise; no §9 MP3 bitrates or low-pass family; ffmpeg and codecs pinned.
  - Selection set S built once from val-A (seed 2026): clean, noise 10 dB, reverberation, Opus 16 kbps, narrowband; hashes stored.
  - Tests on synthetic audio: determinism with a seed, family probabilities, peak limit, disjointness checks.
- **Dates:** Corpora Fri 13 Nov · module and S merged Wed 18 Nov.
- **Gate:** Merged with tests; pins and hashes recorded; Donia's review.

#### V2 · Candidate models — TO DO

- **Tasks:**
  - `cnn-aug` reuses the v1 network (D42); `crnn` and `aasist` as specified in D49 (`aasist` from a reference implementation pinned by commit, F3 waveform input).
  - Unit tests (shapes, one logit, determinism); short compute benchmark → batch sizes in `configs/` before training.
- **Dates:** Merged Fri 20 Nov. Review Khalil and Iheb.

#### V3 · Training and selection on val-A — TO DO

- **Tasks:**
  - Seed 0 per candidate, max 30 epochs, early stopping on the mean val-A EER over S (patience 5); W&B for every run.
  - Winner = lowest mean; ties: fewer parameters, then `cnn-aug`, `crnn`, `aasist`. Seeds 1 and 2 for the winner; bundle seed = median.
- **Dates:** Selection Fri 27 Nov · seeds 1–2 Mon 30 Nov.
- **Gate:** Selection results committed (validation only); every candidate's val-A table reported.

#### V4 · v2 risk policy and v2 Stage 11 — TO DO

- **Tasks:**
  - Protocol §6 method unchanged on clean val-B, bundle seed only (D45 rule).
  - Runner extended to the v2 model set (small PR); one `stage11_test` claim for v2, all three seeds stored.
- **Dates:** v2 Stage 11 Wed 2 Dec.

#### V5 · ModelBundle v2 freeze (F5) — TO DO

- **Tasks:**
  - Assemble and verify F5 with the existing loader; hand over to Iheb (deployment) and Khalil (Stage 13).
- **Dates:** **Fri 4 Dec, 23:59 Tunis: cut-off (D48 (4)).**
- **Gate:** F5 verified before the cut-off, or v2 is excluded from Stage 13 and that is logged.

## 10. Part D — joint external run and analyses (Stages 13–16) · Khalil, Malak

#### 13 · DEEP-VOICE external run (once, guarded) — TO DO

- **Prepare:**
  - Stage 13 runner (D35) for three model sets in one pass: v1 bundle CNN, v2 bundle model, `baseline-v1`; one stored-score file per model with identical columns (D36); D17 filter, D19 per-speaker table, D33, D34 file-level view.
  - Governance allowlist decision before the run (Protocol §8 open item); durable claim beyond Colab (D35).
  - D52 warm-up analysis on simulated calls from val-B (descriptive).
- **Run:** Sat 5–Sun 6 Dec, after F5 or the cut-off. Review by all four.
- **Report:** H2, H8 (paired v1–v2), RQ2 (CNN vs baseline), the live rule's end-of-call accuracy via D34, 87.5% FAKE share and RVC-only scope stated.

#### 14 · Robustness (FoR test only) — TO DO

- **Owner:** Malak; review Donia.
- **Tasks:**
  - Complete §9 grid for `baseline-v1`, v1 bundle CNN and v2 bundle model (one `stage14_robustness` claim each), reusing each model's Stage 11 clean scores (D41).
  - For v2, label conditions: white noise = unseen family, MP3 = seen family / unseen bitrates, bandwidth = seen family (D50 (3)). H4, H6 (v1), H9.
- **Dates:** v1 from Mon 23 Nov (after Stage 11); v2 from Mon 7 Dec.

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
- **Dates:** Built against the mock W2–W3; real bundle after F4.

#### L2 · Live microphone mode (web app) — TO DO

- **Tasks:**
  - Start/Stop, visible listening indicator, consent and decision-support notice (D51); live screen of the chosen design (D55).
- **Dates:** W3–W4.

#### L3 · Chrome extension (Should) — TO DO

- **Tasks:**
  - Design first: authentication against D47 (SameSite=Strict cookie, Origin check) decided and security-reviewed in SDD v2.0.
  - Manifest V3, `tabCapture` after a user click, offscreen document, tab audio routed back to the speakers; same endpoint; tested on WhatsApp Web and Google Meet.
- **Dates:** Prototype W3; WhatsApp Web W4. First item on the cut line.

## 12. Track P — platform and production (P1–P5, Stage 19) · Iheb

#### P1 · Real model in the API — TO DO

- **Tasks:**
  - Replace `MockAnalyzer` with the verified F4 bundle behind the existing analyzer interface; versions in every record (NFR-07, NFR-09).
  - End-to-end tests with the real bundle; later swap to F5 by the deployment rule (D48 (5)).
- **Dates:** Tue 24 Nov (after F4).

#### P2 · PostgreSQL and process pool (D53) — NOW

- **Tasks:**
  - PostgreSQL behind the repository interface for analyses, notes and D47 sessions; migration; CI service container.
  - Bounded process pool (kills timed-out work); `503` + `Retry-After` when full; no broker.
- **Dates:** W1–W2.

#### P3 · Staging and production (D54) — TO DO

- **Tasks:**
  - Hosting provider chosen and recorded (W1); images to GHCR once per `main` commit; staging auto-deploy; production from a release tag with a required reviewer; HTTPS; smoke test and rollback.
- **Dates:** Staging live W2; production W5.

#### P4 · Load test and limits (D53 (3)) — TO DO

- **Tasks:**
  - Reference workload on staging with the deployed bundle; record K, L, `analysis_timeout_s`, `L_MAX` by decision; tmpfs follows K.
- **Dates:** W4.

#### P5 · Interface redesign and usability (D55) — NOW

- **Tasks:**
  - Three directions (Claude prepares) → team vote → implementation keeping §11 fields, NFR-05 wording, axe scans.
  - Decision board; per-window timeline; live screen.
  - Moderated test, 3–5 people outside the team, fixed task script, SUS ≥ 68; findings and fixes logged.
- **Dates:** Vote by Fri 13 Nov · build W2–W4 · test W5.

#### 19 · Integration, report and delivery — TO DO

- **Tasks:**
  - SRS v4.0 (Khalil, W2) and SDD v2.0 (Iheb, W3); README and setup current; fresh-machine test (NFR-10).
  - Final report: each owner writes their stages; demo audio set; two full rehearsals.
- **Dates:** W5–W6; submission target Wed 23 Dec.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| v2 training does not fit the free T4 time | One seed per candidate during selection; 30-epoch cap; batch benchmark in W1; cut line drops `aasist` first; cut-off protects Stage 13. |
| Stage 11 reproduction gate fails (feature bytes differ) | Dry run first on validation splits; a mismatch stops Stage 11 for a logged decision before any test access (D46 (4)). |
| Test results tempt a change to v2 | v2 design approved before any test result (D48 (6)); deployment by rule (D48 (5)); registry refuses reruns. |
| Iheb's tracks are the heaviest | Cut line starts with the extension; Khalil writes SRS v4.0; UI directions prepared by Claude; reviews spread. |
| Extension cannot authenticate safely | Design and security review before code (D51); live mic mode remains the Must. |
| Merge before approval commit | Approval commit on the branch before merge (Section 5.1); record PR otherwise, as #112. |
| AI-generated errors accepted unchecked | Line-by-line review, real test output, CI with torch; ground truth over self-report. |

## 14. Open items this week (W1)

1. Monday sync: confirm every date in Section 7 and the delivery date against the course calendar.
2. Donia: Stage 11 PR B (Claude Code prompt), then PR C.
3. Malak: D50 corpora licence, hashes and split; augmentation module (Claude Code prompt).
4. Khalil: Stage 13 runner for three model sets; start the SRS v4.0 outline.
5. Iheb: hosting provider recorded; PostgreSQL and process pool; review of the three UI directions.
6. All: vote on a UI direction by Friday 13 November.
7. Adopt this plan: one docs PR with this file and its PDF in `docs/reference/`.

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
| D53–D54 | CPU process pool, PostgreSQL, measured limits; dev/staging/production (ADR 0015). |
| D55 | Interface redesign and usability evidence. |
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
| Upload-only service; live detection out of scope | Live mic mode and `/api/live`; Chrome extension (D51, D52). |
| SQLite; one environment | PostgreSQL, process pool, measured limits; staging and production (D53, D54). |
| 14-week targets | Dated six-week plan with a critical path and cut line. |
| Khalil (surname not recorded) | Khalil Dridi; owns SRS v4.0 and the joint Stage 13 run. |

---

*Prepared from the decision register (D1–D55), Protocol v1.0 and its addendum, the project log (Entries 1–154), Execution Plan v2.1, the v2.2 draft, SRS v3.0 and SDD v1.0, with AI assistance (Claude); to be reviewed by the team.*
