# Voice-Spoofing Detection for Fraud Prevention

## Execution Plan v2.2

**Status:** DRAFT FOR TEAM REVIEW - consolidation of existing approved decisions; not a new experiment authorization.

**Prepared:** 3 November 2026. **Snapshot:** `b70dde496bf9367daadc353982a4e1e4bb9e95a8` (merged PR #102).

**Team:** Donia Mabrouk, Iheb Zemzemi, Malak Ben Salem, Khalil Dridi.

**Repository:** https://github.com/kahlanzzi/voice-spoofing-detection

**Version history:** v2.1, 12 October 2026, remains the historical reference. This proposed v2.2 updates the working plan using the existing decision register, protocol and reviewed implementation. It does not replace v2.1 as the adopted reference until reviewed through a team PR. No approval of this draft is inferred from approval of the decisions it summarizes.

### How to read this edition

Read the current status below, the ownership/handoff tables, and the stage you are preparing. Detailed experimental specifications remain in their controlling documents. This plan is an execution guide, not a replacement for the frozen protocol or original evidence. Where an older reference differs from an approved amendment, follow the approved amendment and cite it.

| Topic | Update from v2.1 | Basis |
|---|---|---|
| Frontend | React + TypeScript, Vite, nginx-served `web/`; replaces the Streamlit plan | D38 / ADR 0008 |
| API | Current `/api` contract, analysis/history/notes/stats and session routes; old `/analyze-audio` example is no longer the current route | API design, PR #65 and subsequent reviewed changes; D47 |
| External evaluation | Both frozen models in the same single pass; dataset-wide claim and durable registry evidence | D33-D36 / ADRs 0007 and 0009 |
| Internal test and robustness | Closed registry purposes; stored clean scores reused for paired robustness comparisons | D31 amended by D41 / ADR 0010 |
| Model preparation | Frozen F3, baseline fit, CNN training/selection and risk-fit decisions replace v2.1's unresolved options | D37, D39, D40, D42, D44, D45 |
| Team workflow | Stage briefs and teach-back are part of stage closure; original D26 ownership is retained | D26, D32 |
| Progress | Stages 0-10 closed; Stage 11 current; more backend functionality merged, but real-bundle integration and React delivery not complete | Pinned main snapshot and open PRs #67/#68 |
| Navigation | Correct repository URL and current folder descriptions; v2.1 and original references remain unchanged | README, architecture and docs index |

### Current snapshot - plans are not completed implementation

- **Research:** Stages 0-10 and F1-F3 are recorded complete. Stage 11 has not been declared executed by this plan. D46 runtime-plan PR #67 and runner PR #68 remain open at the checked snapshot; their proposals are not adopted here.
- **Backend:** PRs #77-#85 add analysis intake with mock scoring and SQLite, history, notes, stats, sign-in and session/Origin protections. PRs #86-#87 add the API container and nginx/Compose deployment setup.
- **Remaining backend:** integrate the real ModelBundle after F4 and verify production deployment and integrated acceptance.
- **Frontend:** React/Vite and the web image landed in #88-#90. Sign-in, the app shell, theme controls, System status and shared UI components landed through #102. Upload PRs 8b-8d remain follow-up at this snapshot.
- **Documentation:** backlog, UML, AI ethics reflection, Part 6 report/evidence, README refresh and sectioned docs index are committed. Final course formatting, demo, user guide, signatures and submission must be confirmed separately.

<!-- pagebreak -->

## 1. Purpose, authority and scope

### 1.1 Purpose and research questions

Build a decision-support system for short speech recordings: authentic, synthetic or uncertain, with confidence and LOW/MEDIUM/HIGH risk information for an analyst. Primary deployment scenario: bank call-center triage of high-risk requests; secondary: executive voice messages. A result is not proof of fraud or a biometric identity claim.

Retain the protocol's four research questions: FoR-to-DEEP-VOICE generalization; CNN versus baseline in-domain and externally; robustness under predefined degradations; and calibration under dataset shift. Preserve H1-H7 and their original tests/readings. Do not reinterpret an observed validation result as a new preregistered hypothesis.

### 1.2 Controlling documents

The existing priority order is unchanged: approved decision register > SRS v3.0 > SDD v1.0 > Execution Plan > Proposal. Protocol v1.0 and approved amendments govern execution and freeze/change control. An unresolved conflict must be stated and reviewed; this draft cannot silently resolve or approve it.

The original SRS/SDD/proposal/v2.1 files remain historical references. The React and session amendments are applied through their recorded decisions and reviewed contract. Numeric API values still marked proposed or open in the contract remain proposed/open even if a provisional implementation exists.

### 1.3 Scope lock

- Baseline: MFCC mean/std features and RBF-SVM, evaluation only (D7, D37). The deployment API never loads this baseline as an alternative model.
- Deployed model: one CNN on log-Mel features, with the frozen preprocessing profile and risk policy. No extra architectures, ensembles or alternative features.
- Existing stretch work: robustness and calibration/reliability analysis, under the frozen protocol. No augmentation in model v1.
- Online tier: FastAPI plus the approved React/TypeScript analyst interface. The interface calls the API and contains no detector or risk-policy logic.
- Out of scope: streaming, replay/adversarial-defense extensions, explainability, extra datasets/models and speculative additions to the prototype. The old parking-lot ideas are not authorized work.

### 1.4 Non-negotiable experimental safeguards

1. DEEP-VOICE is used for scoring exactly once, after F4 and the reviewed Stage 13 prerequisites. Before then, D20 permits metadata-only activities: download/archive hash, extraction, path/size/byte-hash listings and the metadata audit. No listening, spectrograms, feature statistics, tuning or exploratory scoring.
2. FoR test access is governed too. Use the approved Stage 11/14 purposes and reviewed implementation. A dry run or a renamed purpose is not an exception.
3. Fit models on FoR train, make training/model decisions on val-A, and fit calibration/threshold/band on val-B as specified. No external/test feedback may tune the system.
4. Freeze the CNN checkpoint, profile and risk policy together before external access; verify the baseline artifact separately for the external comparison.
5. Preserve dataset pins, manifests, preprocessing, metrics and thresholds. Any required change follows the existing decision/change-control process before affected results are seen.
6. Never commit audio, datasets, checkpoints, credentials or tokens. This document was prepared from repository records without reading protected audio or executing a model.

### 1.5 Metrics and limits

Spoof is the positive class. Report EER/ROC-AUC at score level, and applicable accuracy, precision, recall, F1, class counts, false-positive/false-negative rates, three-way confusion matrix and coverage/deferral at the frozen decision policy. Include confidence intervals and denominators; never accuracy alone or t-DCF. The baseline has its own D29 forced-binary threshold, no calibrated probability, uncertain band or risk policy. External baseline reporting follows D36.

Use the existing bootstrap conventions. Undefined estimates remain undefined; invalid resamples are excluded and counted as `n_invalid`, reported with each interval. Speaker-clustered external intervals have only eight source-speaker clusters; report that limitation and the 87.5% FAKE window imbalance. Describe transfer to this RVC dataset, not all voice cloning or a causal explanation of any gap.

<!-- pagebreak -->

## 2. Ownership, reviews and handoffs

The D26 allocation is retained. A stage owner drives implementation, evidence and questions; ownership does not replace teammate review or allow unilateral changes to frozen interfaces.

| Member | Responsibility | Consumer/review relationship |
|---|---|---|
| Donia Mabrouk | Part A foundation; Stage 3 lead; Part C, Stages 7-12 | Part B consumer; Khalil/Iheb consume Part C; protocol sign-off by all four |
| Malak Ben Salem | Part B, Stages 4-6; Stage 14 robustness | Donia reviews Part B; approved protocol also names Khalil for Stage 14 review |
| Khalil Dridi | Stage 13 external evaluation; Stage 15 reliability; Stage 16 error analysis | External results reviewed by all four; Donia reviews analysis, with stage-specific protocol reviewers retained |
| Iheb Zemzemi | Part E, Stages 17-19: API, React, integration and delivery | Malak is the Part E consumer reviewer; technical contract reviewers remain as recorded |
| All four | Stage 3 sign-off, Stage 13 review, Stage 19 report/demo/defense | No author self-approval; each member can explain the full pipeline |

The broad plan reviewer allocation does not remove additional reviewers explicitly named in the protocol, CODEOWNERS or a particular decision.

| Handoff | Contract to preserve |
|---|---|
| B -> C: Malak -> Donia | Manifest fields/meaning, immutable F2 identities and F3 preprocessing I/O. Protocol uses the SHA-256 manifest field; do not rename columns based on this summary. |
| C -> D: Donia -> Khalil | Verified ModelBundle, scoring semantics, immutable baseline identity, frozen D29/risk thresholds, and approved stored-score formats. D36 adds the external baseline file without adding the baseline to deployment. |
| C -> E: Donia -> Iheb | Verified bundle loader and versions; success contract: prediction, confidence, risk_level, model_version, profile_version, policy_version, request_id, notice. Preserve the approved route-specific response envelope. |
| B -> Stage 14: Malak | Approved waveform degradations before unchanged F3; pairing with stored Stage 11 clean observations under D41. |

### Freeze points and closure

| Freeze | Stage | Meaning |
|---|---|---|
| F1 | 3 | Experimental Protocol v1.0 and recorded change control |
| F2 | 5 | Dataset split/manifests/hash identities |
| F3 | 6 | One deterministic preprocessing profile, reused by research and inference |
| F4 | 12 | Immutable ModelBundle plus recorded baseline artifact hash for D36 |

A stage closes only after its required evidence, recorded review and gate are complete, the work is merged with required CI, and the project log/current status are updated. D32 additionally requires its stage brief and accepted teach-back. Merge of helper code alone does not mean the corresponding real-data run or freeze happened.

The dependency chain remains F1 -> split/F2 -> preprocessing/F3 -> modeling/selection/risk -> Stage 11 -> F4 -> single external run -> analyses. API/UI scaffolding may proceed in parallel against mocks. Real-bundle inference waits for the approved handoff and F4; no demo need can waive a research gate.

<!-- pagebreak -->

## 3. Roadmap at the pinned snapshot

Status is evidence-based at `b70dde4`; no new completion is asserted beyond that snapshot. The old semester-week targets were planning guidance, not evidence of completion. This edition does not invent revised deadlines; owners must agree remaining dates against the course calendar.

| Stage | Work | Owner | Snapshot status / closure evidence |
|---|---|---|---|
| 0 | Repository and environment | Donia | Closed; workflow, CI and pinned dataset setup recorded |
| 1 | Dataset audit | Donia | Closed; full FoR and metadata-only external audit |
| 2 | Threat model | Donia | Closed; reviewed threat model and limitations |
| 3 | Protocol F1 | Donia + all | Closed; four-member sign-off and accepted brief |
| 4 | Leakage-aware split | Malak | Closed; group/stratification checks |
| 5 | Dataset freeze F2 | Malak | Closed; manifest/hash evidence |
| 6 | Preprocessing F3 and benchmark | Malak | Closed; profile, invariance/shortcut and benchmark evidence |
| 7 | Harness and MFCC/SVM baseline | Donia | Closed; baseline validation evidence and brief |
| 8 | CNN training | Donia | Closed; training/validation evidence |
| 9 | Model selection | Donia | Closed; selected configuration/seeds and brief |
| 10 | Risk-policy fit | Donia | Closed; val-B risk evidence and accepted brief |
| 11 | FoR internal test once | Donia | Current; D46 proposal #67 and runner #68 still open; no run completion claimed |
| 12 | ModelBundle freeze F4 | Donia | Pending Stage 11 and reviewed freeze; verifying loader already exists |
| 13 | External evaluation once | Khalil | Gated by F4 and reviewed runner; approved reporting decisions are not access permission |
| 14 | FoR robustness | Malak | Planned; D41 prerequisites and registry/storage rules apply |
| 15 | Reliability analysis | Khalil | Planned from authorized stored scores; no recalibration |
| 16 | Error analysis | Khalil | Planned; H5-H7 and D43 as specified |
| 17 | FastAPI and deployment | Iheb | API routes, session protections and container setup merged; real-bundle integration and deployment acceptance pending |
| 18 | React analyst interface | Iheb | In progress; React sign-in, shell and System status merged; upload flow remains follow-up |
| 19 | Tests, documentation, delivery | Iheb + all | Ongoing; documentation merged, full integrated acceptance/demo/submission not established |

### What v2.2 does not infer

Validation results do not demonstrate test/external performance. A mock API response does not show that a frozen bundle is loaded. Existing backend tests do not show the final browser flow works. Passing CI and merged documents do not establish a complete midterm MVP, signed honesty statement, accessible demo or LMS submission.

Current status is primarily checked from merged files and PR records; it may advance while this draft is reviewed. Update the snapshot explicitly rather than silently changing an older record or relabelling an unmerged branch as complete.

<!-- pagebreak -->

## 4. Part A and Part B - foundation, data and preprocessing

### Stage 0 - repository and environment

Maintain the protected main/PR process, Python 3.13 official environment, dependency pins, shared hooks, tests and CI. Python 3.14 may be used locally as permitted by the repository. Establish account access and approved data/compute locations without committing credentials. New-member setup is documented in the onboarding section, not a reason to reopen frozen data.

### Stage 1 - dataset audit

Preserve the signed audit and existing evidence, including label balance, formats, near-duplicate grouping, MP3-history and bandwidth/silence limitations. FoR is the train/validation/internal-test dataset; DEEP-VOICE is the external RVC dataset and remains metadata-only until Stage 13. Detailed counts, hash identities and limitations remain in the audit/manifests, rather than being recomputed for this document.

### Stage 2 - threat model

Keep the reviewed scenario, threats T1-T7 and mappings to experiments, misuse/oracle concerns, privacy controls and residual risks. Interface wording must communicate decision support and uncertain outcomes. A UI demonstration does not validate security controls or justify a stronger claim about fraud detection.

### Stage 3 - protocol F1

Keep the signed protocol and amendments as the executable specification for research questions, hypotheses, split, metrics, policy, external windows/filtering, degradation grid, interfaces and freezes. Each owner remains responsible for the section they execute. Do not create a fresh protocol or change frozen values merely to synchronize plan wording.

### Stage 4 - leakage-aware split

Use near-duplicate connected groups at similarity 0.95 with no group crossing splits. Retain REAL/FAKE_mp3/FAKE_other stratification and the protocol's group-aware split and seed. Train/validation/test are 70/15/15; validation is divided into val-A and val-B. Preserve existing checks/reporting for group isolation, split sizes and subgroup proportions. The lack of speaker metadata limits the interpretation of in-domain performance.

### Stage 5 - dataset freeze F2

Retain the committed FoR and metadata-only external manifests and their hash ledger. Do not regenerate or rename frozen columns/identities while editing this plan. A dataset pin, sample identity, label, split or grouping change requires an approved decision and the relevant change-control process.

### Stage 6 - F3 preprocessing and compute evidence

D39 specifies channel-mean mono, `soxr_hq` resampling to 16 kHz, deterministic 20 ms RMS edge trimming at -50 dBFS, then centered pad/crop to 32,000 samples; preserve all-silent inputs. Use the exact frozen implementation/config for edge cases and filter speech fractions. No new VAD choice or normalization is opened by v2.2.

The CNN log-Mel tensor is (1, 64, 197); the baseline has a 40-value vector from mean/std of 20 MFCCs. The trained CNN's train-only Mel standardization is part of its checkpoint under D42, not an ad hoc change to F3. Apply this same frozen path in notebooks, test runners and production inference.

Preserve invariance and shortcut evidence and the existing compute benchmark. The original plan's suggestion of a brief training benchmark is not permission to rerun training or rewrite evidence: the recorded Stage 6 compute dry run had no optimizer step. Use the committed results and their stated limitations to justify resource choices. F3 and the Stage 6 brief are recorded closed.

<!-- pagebreak -->

## 5. Part C - modeling, internal test and freeze

### Stage 7 - shared harness and baseline

Use the existing reusable evaluation harness with synthetic known-value tests. Baseline D37 uses MFCC features standardized with train-only mean/std, RBF-SVM fitting/selection on the predefined grid using val-A, and the stored deterministic artifact. Its score orientation is higher = more likely spoof. D29 fits its own val-B EER-point threshold; it is forced binary only and remains evaluation-only.

Correcting v2.1: D40 exempts the deterministic Stage 7 baseline fit from W&B; preserve its committed provenance, reports, cache bindings and artifact hash. Do not add logging retroactively or regenerate evidence to match older wording.

### Stages 8-9 - CNN training and selection

Retain the D30 search space: 3/4 convolution blocks and learning rates 1e-3/3e-4, Adam, maximum 30 epochs and val-A EER early stopping with patience 5. D42 governs exact blocks/head, train-only per-Mel standardization, loss, checkpoint format and early-stopping/tie behavior. The selected configuration is evaluated across seeds 0, 1 and 2 under the existing rule; model selection uses val-A, never the test or external split.

Log every CNN training/selection run, including quick tests, to W&B according to D40/D42. No audio, external/test data, per-clip outputs or val-B material is uploaded there. Preserve the committed validation results as validation-only evidence and retain the shortcut/near-perfect-performance caveats.

### Stage 10 - fit the frozen policy on val-B

Preserve the selected bundle seed and the existing risk-v1 fit: calibration, EER-point operating threshold, uncertain band and LOW/MEDIUM/HIGH decision mapping. D44 specifies the approved Platt stopping/guard behavior. Do not reopen the old choice between calibration methods. Preserve exact boundary handling and the fitted policy artifacts.

D45: only the bundle-seed CNN has a fitted policy. Other CNN seeds do not receive separately fitted probabilities/predictions for Stage 11. No test/external scores may be used to fit or adjust calibration, thresholds or deferral.

### Stage 11 - one FoR test run

Owner Donia; use the existing protocol, D31/D41 registry and approved storage implementation. The registry key is `(dataset, split, evaluation_purpose, model_set)` with closed purposes `stage11_test` and `stage14_robustness`; arbitrary purpose names are refused. D41 retains clean per-clip identities, labels, D14 groups and raw scores for the baseline and all three CNN seeds.

Store probability/prediction for the bundle-seed CNN, baseline D29 prediction with calibrated p not applicable, and raw scores for the other CNN seeds with probability/prediction not applicable (D45). Report applicable D4 metrics and preregistered subgroup tests. Keep class counts, confidence intervals, binding hashes and run identity with outputs.

At this snapshot, D46 is still proposed in PR #67 and runner implementation is open in PR #68. Their proposed runtime, rehearsal, environment and durable-run details are not silently approved by v2.2. Obtain the required approvals and implementation review before any FoR test read. This document neither executes nor authorizes that read.

### Stage 12 - ModelBundle freeze F4

After the required Stage 11 work/reviews, assemble the immutable checkpoint + F3 profile + risk policy and manifest, with component version/hash verification. D36 also records the separate baseline artifact hash in the F4 freeze record; it is not an extra deployment model. The verifying loader's existence does not mean F4 has occurred. Record the freeze through the required reviewed process before Stage 13.

<!-- pagebreak -->

## 6. Part D - single external evaluation and analysis

### Stage 13 - one guarded external run, both frozen models

Owner Khalil; all four review. D35/D36 require verification of both the ModelBundle and the separately frozen baseline artifact before any audio read or model call. A single dataset-wide claim covers both models on identical windows in the same pass; changing the model set does not permit another external run.

The claim must survive the notebook/session: commit it to the repository or write it to Drive and verify durability before the first audio read. Immediately after the run, record the registry SHA-256 and line count in the project log. Preserve the D31 truncation lesson; a text allowlist is not a substitute for actual runtime/I/O safeguards.

Split files into non-overlapping complete 2-second windows from file start; drop the short remainder as specified. Apply the same frozen preprocessing separately to every window, including trim and centered padding. Required integration fixtures use synthetic audio/fake scorers and fail if preprocessing is applied to the whole file before splitting. Use the final F3 API; do not duplicate it.

Derive filtered primary and unfiltered sensitivity views from this one run. The existing filter uses the frozen detector/speech-frame definition and threshold from F1/F3; report filter exclusions separately from model deferrals. Save the approved per-model score outputs, provenance, expected windows and manifest/run identities so later analysis uses stored scores without reopening external audio.

Report pooled metrics, all eight source-speaker rows and the declared speaker average. Under D33, undefined rates propagate in the eight-speaker mean; retain expected speakers, defined counts and reasons. Do not omit an undefined speaker or replace an unavailable rate with zero. Report `n_invalid` beside each interval.

D34's secondary file-level CNN decision is the mean of calibrated window probabilities per file/view, followed by the unchanged frozen threshold/band. Do not average raw scores then calibrate, average hard labels, tune new thresholds or mix filtered/unfiltered windows. Explicitly report files with no evaluable windows. File-level intervals still cluster by source speaker.

The baseline uses its frozen D29 forced-binary threshold with no calibration/band/risk output. Apply D36's reporting rules and the paired CNN-minus-baseline EER comparison with shared speaker-bootstrap resamples. The baseline is not the source of the CNN's file-level calibrated decision. Report the RVC-only interpretation, imbalance and eight-cluster limitations even if results look excellent or poor.

D35 permits only the exact reviewed runner/config additions, not broad allowlist directories. Approved ADRs alone grant no audio access. The reviewed runner, governance implementation, F4 and the Stage 13 review gate remain prerequisites; this document adds none of those permissions.

### Stage 14 - FoR robustness only

Owner Malak. Use D41's single `stage14_robustness` claim per frozen model set for the complete predeclared grid. Apply noise, compression and bandwidth degradations to the waveform before unchanged F3. The protocol levels remain: white-noise SNR 30/20/10 dB; MP3 128/64/32 kbps; low-pass 6/4/3.4 kHz. Do not create robustness runs on DEEP-VOICE.

Reuse Stage 11 clean scores; do not make a second clean pass. Pair by stable clip ID, use the frozen bundle-seed CNN and baseline, and preserve the specified D14-group bootstrap (2,000 resamples, seed 2026). Retain the H6 genuine-caller endpoint and the D41 paired baseline EER endpoint/read rule. No fitting, recalibration, threshold change or model selection follows these results.

### Stages 15-16 - reliability and errors

Owner Khalil. Use authorized stored FoR/external scores for 10-bin equal-width reliability diagrams/ECE and uncertainty-under-shift analysis; evaluation is not a new calibration fit. Analyze errors by subgroup, source speaker, confidence and deferral, tied to threats T1-T7 and H5-H7.

D43 adds the predefined within-class loudness association analysis on stored Stage 11 outputs: use its exact frozen feature/statistical procedure; no new model inference or evaluation purpose. State exploratory/association limits and never infer causation or tune the model from this diagnostic. Read external scores only; do not rerun models or reopen external audio for later reports.

<!-- pagebreak -->

## 7. Part E - FastAPI, React and delivery

### Stage 17 - API and inference integration

Owner Iheb. The approved API design, frozen success fields and subsequent D47 session contract replace v2.1's illustrative `POST /analyze-audio`. All routes below are under `/api`; route presence does not imply real model inference or a complete security rollout.

| Method | Path | Purpose / snapshot note |
|---|---|---|
| GET | `/status` | Analyzer/service versions and readiness; preserve explicit mock indication |
| GET | `/limits` | Accepted formats, upload limits and deletion statement from approved configuration |
| POST | `/analyses` | Intake/validation/scoring and persistence; merged implementation defaults to MockAnalyzer |
| GET | `/analyses` | Filtered/paged history |
| GET | `/analyses/{request_id}` | Stored result/detail |
| PUT | `/analyses/{request_id}/note` | Analyst note; account attribution integration still requires verification |
| GET | `/stats` | Operational stored-record statistics, never ground-truth quality metrics |
| POST/GET/DELETE | `/session` | Sign-in / current account / sign-out under D47 |
| GET | `/health` | Liveness, no sensitive data |

The real analyzer must load/verify the frozen bundle and reuse shared F3 and risk code. Until that handoff is complete, keep mock identities/output explicitly labelled. Preserve request IDs, version binding, limits, timeout/error shapes, privacy, retention and refused-input behavior. Never expose a prediction in an error or fabricate real-model readiness from a mock response.

### D47 security target versus merged implementation

D47 records username/password sign-in with admin-created accounts, argon2id password hashes, server-side SQLite sessions and hashed session IDs. Its target includes Secure/HttpOnly/SameSite cookie handling, Origin checks for state changes, username lockout, nginx rate limiting and note-author attribution. Session lifetimes and other values still labelled proposed in the contract are not made final here.

PR #84 adds session/health endpoints and startup checks. PR #85 extends session and Origin enforcement and adds note-author attribution. PRs #86-#87 add container and proxy setup. Verify HTTPS, cookie and proxy behavior under the intended deployment before claiming production readiness.

### Stage 18 - React + TypeScript analyst interface

D38/ADR 0008 replaces Streamlit. Use the approved strict TypeScript/Vite app in `web/`, plain CSS/design tokens, pinned Node/toolchain and committed lockfile. Serve static output via the nginx `web` container, proxying API calls. The app has no detector, threshold, calibration or independent statistics logic.

The merged React app has the sign-in screen, shell, theme controls and System status. PR #102 adds shared controls. The upload flow is split into follow-up PRs 8b-8d. Complete Analyze and History before the lower-priority Overview. Overview uses API operational figures only. Preserve decision-support wording, accessibility and the distinctions between verdict, risk and confidence.

Test against contract-faithful mocks while real inference is gated. Require TypeScript checks, frontend lint/unit tests and browser end-to-end tests of the approved flow before final acceptance. The HTML prototype is a design artifact, not the React application. D38's fallback to Streamlit requires its own decision; passing a calendar week alone does not activate it. The existing v1 Analyze-flow/week-10 fallback rule is retained as a review checkpoint, not silently exercised here.

### Stage 19 - integration, report and delivery

Complete reviewed API/web containers, configuration and documented startup, then independently execute a fresh-machine/clean-run acceptance test. Verify valid input, invalid/empty/oversized files, failures, session expiry and the displayed verdict/confidence/risk/notice against the actual backend. Tests must not depend on a protected research run.

Assemble the existing required proposal/requirements/UML/report/AI-log artifacts and required PDF exports, an end-user guide, signed academic honesty statement and accessible demo. Include all four names and confirm instructor repository/video access. Follow the course checkpoint instructions; a full-project roadmap is not a claim that every stage is required before the midterm.

Demonstrate only permitted material. No protected dataset audio is used just to make a demo; external examples cannot precede the authorized registered run, and any later disclosure still needs the existing data/privacy permissions. Label mock versus real behavior. Rehearse the presentation and each member's explanation of their own contribution and the shared pipeline.

<!-- pagebreak -->

[Stage 18 sign-in](https://github.com/kahlanzzi/voice-spoofing-detection/pull/95) | [System status](https://github.com/kahlanzzi/voice-spoofing-detection/pull/101) | [Shared components and upload follow-up](https://github.com/kahlanzzi/voice-spoofing-detection/pull/102)

## 8. Parallel preparation, workflow and reproducibility

### Work that can proceed before the next gate

| Owner | Useful independent preparation | Boundary |
|---|---|---|
| Donia | Stage 11 review/reproduction preparation and synthetic rehearsals under the agreed plan | No FoR test execution before required reviews/approvals |
| Malak | Predeclared degradation tools/tests, UML review and midterm/editorial coordination | Preserve F3; no unauthorized test/external data |
| Khalil | Stage 13/15/16 reporting and completeness tests with synthetic or permitted validation scores; contract/acceptance cases | No external audio or official result claims from synthetic fixtures |
| Iheb | API/React mock integration, response/schema checks, browser/accessibility tests and deployment preparation | Real bundle after reviewed F4 handoff; coordinate shared-file ownership |
| All four | Independent setup verification, stage teach-backs, AI disclosures, user-guide/demo drafts and submission checks | Report unverified work honestly; no secrets or protected audio in shared artifacts |

### PR and documentation workflow

Use one focused D23 branch/PR and Conventional Commits. Main remains protected with required review and CI. Stacked branches must be retargeted to main after their predecessor merges before merging, unless a different base is explicitly intended. AI agents never merge or approve their own output; humans review and accept it.

Before publishing, run the appropriate full suite and governance checks, scan for secrets/data, and preserve actual output. Update CHANGELOG and append the project log; never rewrite old entries. Check main and open PR/branch reservations before claiming log, D-ID or ADR numbers. D32 stage briefs and teach-backs remain part of closure. Regenerate the README inventory when files are added or moved.

AGENTS.md remains the shared AI rulebook; tool-specific files point to it. Label verified facts, document-derived statements and hypotheses. Disclose the AI tool and task in the PR/log. A generated answer, PDF or test report is not human approval. Existing tool permissions/policies are not expanded by this execution-plan refresh.

### Data, environment and reproducibility

Keep datasets in the approved shared Drive/compute locations, never Git or personal workspaces outside the existing data policy. Use pinned archive/revision/hash identities from configuration and the hash ledger. Copy to authorized local compute storage before heavy I/O; do not introduce a new hardcoded path into shared code.

Every official result must retain dataset identity, config, code revision, environment and run/provenance records. Preserve D40's distinction between baseline committed evidence and CNN W&B logs. Use the official Python version and current pinned dependency/config instructions; CPU test runs and GPU experiments are different environments and must not be presented as comparable performance measurements.

For onboarding, read this plan with AGENTS/CONTRIBUTING, current log, audit/limitations, threat model and relevant protocol/decisions. Obtain approved account/Drive access without sharing secrets. Run the repository setup and tests, verify the AI tool loads the project rules, make a small reviewed contribution and explain the single-use external rule and assigned handoff to a teammate. No onboarding task authorizes data inspection or a new research run.

<!-- pagebreak -->

## 9. Risks, remaining work and review gate

| Risk | Existing mitigation / review action |
|---|---|
| Test/external leakage or repeat scoring | Frozen protocol, reviewed runtime guards/registry, D20 restrictions and governance tests; inspect actual I/O too |
| Dataset shortcuts or overclaimed transfer | Existing subgroup tests, H5-H7, D41/D43 diagnostics, per-speaker counts/CIs and stated RVC/metadata limitations |
| Research/online preprocessing drift | Shared F3 implementation, verification at bundle load and per-window integration tests |
| AI or stale-document claims treated as evidence | Human PR review, exact outputs, pinned snapshots, no silent source/conflict resolution |
| Mock prototype mistaken for working model | Explicit mock identities and demo labels; real-bundle and integrated acceptance gates |
| Frontend scope delays core flow | D38 Analyze/History priority; Overview last; fallback only through the existing decision process |
| Session/deployment security incomplete | D47 target, current contract, deployment tests and explicit distinction from partial implementation |
| Dependency or compute mismatch | Recorded environments and Stage 6 resource evidence; no guessed speedups or unapproved reruns |
| Fragmented team understanding | Consumer review, accepted D32 briefs, teach-backs and joint demo rehearsal |

### Remaining actions at this snapshot

1. Review this consolidation for accuracy; retain D26 ownership and D32 handoffs. Do not mark v2.2 adopted until the team PR is accepted.
2. Complete the required review/merge of D46 #67 and Stage 11 runner #68 before any official internal-test access; then follow the approved Stage 11/F4 sequence.
3. Complete the reviewed Stage 13 runtime/completeness/artifact-verification implementation before the single external run. Approval of D33-D36 is not execution permission.
4. Finish backend guard/note-author work, real-bundle handoff when available, React implementation, containers and integration acceptance. Reconcile status with newer PRs rather than copying a stale snapshot.
5. Independently validate startup/demo/user-guide material, required PDF formats, honesty signatures and submission access. Keep final editorial/course acceptance separate from implementation PR approvals.

### Adoption checklist - intentionally unapproved

| Reviewer | Review responsibility | v2.2 acceptance |
|---|---|---|
| Donia Mabrouk | Protocol, decisions, model/testing/freeze sequence | Pending |
| Malak Ben Salem | Data/F3/robustness, workflow and handoffs | Pending |
| Khalil Dridi | External governance, reporting and source comparison | Pending |
| Iheb Zemzemi | React/API/session/deployment targets and implementation status | Pending |

Record actual review dates and PR evidence after review; do not infer acceptance from silence, previous ADR approval or CI. If review changes a requirement/interface/governed rule, use the existing decision process instead of disguising it as an editorial v2.2 update. Once adopted, update reference/index pointers and record what v2.2 supersedes; retain v2.1 and frozen documents unchanged.

### Verification and AI disclosure

Prepared with Codex from v2.1 and pinned repository documents/code/PR metadata. Checked scope: all 20 stage numbers and D26 owners retained, current approved amendments distinguished from open proposals, links and PDF rendering. No model, dataset, registry, code or frozen evidence was modified; no new runtime test or real-data measurement is claimed. The team still reviews factual accuracy, completeness and adoption.

<!-- pagebreak -->

## 10. Sources and traceability

All links below are pinned to the checked main commit unless they intentionally point to an open PR. These are human-readable sources for the consolidation, not a claim that every implementation acceptance test has been rerun.

| Source | What it controls or supports |
|---|---|
| [Historical execution plan v2.1](https://github.com/kahlanzzi/voice-spoofing-detection/blob/2e42e1b7aa9979135003124148faa15afea2d426/docs/reference/Execution_Plan_v2.1.pdf) | Original ownership, 20-stage roadmap, handoffs and onboarding |
| [AGENTS.md](https://github.com/kahlanzzi/voice-spoofing-detection/blob/2e42e1b7aa9979135003124148faa15afea2d426/AGENTS.md) and [CONTRIBUTING](https://github.com/kahlanzzi/voice-spoofing-detection/blob/2e42e1b7aa9979135003124148faa15afea2d426/CONTRIBUTING.md) | Priority, hard rules, reviews and workflow |
| [Decision register](https://github.com/kahlanzzi/voice-spoofing-detection/blob/2e42e1b7aa9979135003124148faa15afea2d426/docs/decisions/README.md) | Exact approved D-IDs and qualifications; proposed/open statuses remain visible |
| [Protocol v1.0](https://github.com/kahlanzzi/voice-spoofing-detection/blob/2e42e1b7aa9979135003124148faa15afea2d426/docs/experimental_protocol.md) | Hypotheses, split, metrics, policy, windows/filter, degradation levels and F1-F4 |
| [SRS v3.0](https://github.com/kahlanzzi/voice-spoofing-detection/blob/2e42e1b7aa9979135003124148faa15afea2d426/docs/reference/SRS_Voice_Spoofing_Detection.pdf) and [SDD v1.0](https://github.com/kahlanzzi/voice-spoofing-detection/blob/2e42e1b7aa9979135003124148faa15afea2d426/docs/reference/SDD_Voice_Spoofing_Detection_Lab5.pdf) | Requirements and original design, subject to approved amendments |
| [ADR 0007](https://github.com/kahlanzzi/voice-spoofing-detection/blob/2e42e1b7aa9979135003124148faa15afea2d426/docs/decisions/0007-stage13-reporting-and-access.md) / [ADR 0009](https://github.com/kahlanzzi/voice-spoofing-detection/blob/2e42e1b7aa9979135003124148faa15afea2d426/docs/decisions/0009-baseline-in-external-run.md) | D33-D36 undefined/file views, durable dataset claim and external baseline comparison |
| [ADR 0008](https://github.com/kahlanzzi/voice-spoofing-detection/blob/2e42e1b7aa9979135003124148faa15afea2d426/docs/decisions/0008-web-frontend-react.md) | Approved React replacement, scope/order and conditional fallback |
| [ADR 0010](https://github.com/kahlanzzi/voice-spoofing-detection/blob/2e42e1b7aa9979135003124148faa15afea2d426/docs/decisions/0010-stage14-purpose-key-and-paired-scores.md) | D41 closed registry purposes and clean-score reuse |
| [ADR 0011](https://github.com/kahlanzzi/voice-spoofing-detection/blob/2e42e1b7aa9979135003124148faa15afea2d426/docs/decisions/0011-analyst-sign-in-sessions.md) and [API design](https://github.com/kahlanzzi/voice-spoofing-detection/blob/2e42e1b7aa9979135003124148faa15afea2d426/docs/api_design.md) | D47/session target, routes, response envelopes and remaining open decisions |
| [API factory](https://github.com/kahlanzzi/voice-spoofing-detection/blob/2e42e1b7aa9979135003124148faa15afea2d426/api/main.py) / [PR #84](https://github.com/kahlanzzi/voice-spoofing-detection/pull/84) | Default mock analyzer, startup requirements and merged session integration snapshot |
| [Project log](https://github.com/kahlanzzi/voice-spoofing-detection/blob/2e42e1b7aa9979135003124148faa15afea2d426/docs/project_log.md), [stage briefs](https://github.com/kahlanzzi/voice-spoofing-detection/tree/2e42e1b7aa9979135003124148faa15afea2d426/docs/stages), [backlog](https://github.com/kahlanzzi/voice-spoofing-detection/blob/2e42e1b7aa9979135003124148faa15afea2d426/docs/midterm/backlog.md) | Dated evidence, closure and requirement traceability; stale individual status lines are checked against actual merged PRs |
| [D46 PR #67](https://github.com/kahlanzzi/voice-spoofing-detection/pull/67) / [runner PR #68](https://github.com/kahlanzzi/voice-spoofing-detection/pull/68) | Open proposals at the snapshot; not adopted as approved decisions in v2.2 |

**Source-preservation note:** v2.1, original references, protocol, registry, datasets and original experiment outputs were not edited in making this draft. Changes in the v2.2 summary must be reviewed against the controlling sources above before adoption.
